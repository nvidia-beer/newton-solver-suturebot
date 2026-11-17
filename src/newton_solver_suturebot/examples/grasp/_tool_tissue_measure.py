# SPDX-License-Identifier: Apache-2.0
"""Measurements of the tool/tissue example: one record per frame, one summary per phase.

The records are made from the observation of a frame (``ToolTissueScene.observe``), which the
example calls after the step; nothing here touches the device. "Depth" is the depth of the
deepest point of a drawn tool part inside the tissue: the largest distance by which a point of
the part's contact spheres lies inside the surface, to the nearest surface point (negative: the
gap to it). On a flat surface it is the vertical depth below it.

"Capsule depth" is the same question for the exact drawn capsule and not for its chain of
spheres, whose surface lies up to 0.13 radii inside the capsule between two centres: how far the
deepest point of the capsule is inside the closed tissue surface (zero: nowhere): the radius of
the capsule less the distance of its axis from that surface, and for a capsule whose axis is
under the surface the deepest of the points on its own surface (ray-parity sign). It is what the
remainder of a contact is judged by. With
:meth:`Measurements.add_interval` it is recorded at the end of every physics interval.
"""

import json
import math

import numpy as np

from ..._src.solvers.fem.node_holds import HOLD_FACTOR
from ._tool_tissue_path import PHASES, PINCH, _upper_surface_nodes

# Rest tolerance of tissue that a contact has let go [m]: the acceptance number of the swept contact for tissue
# that can yield (its remainder).
RELEASE_TOLERANCE_M = 2.0e-5
# Where the tent around a pinch is read: distances from the pinch site at which the rise of the upper surface is
# taken, and the half-width of the ring of nodes about each [m]. Numbers of the check, in mesh spacings 2.3, 3.8
# and 0.4.
TENT_RINGS_M = (0.003, 0.005)
TENT_RING_HALF_WIDTH_M = 0.0005
# Segments of the pinch in which the jaws are commanded closed and stand still: a slip there is reported.
PINCH_CLOSED_SEGMENTS = ("squeeze hold", "lift", "lift hold", "lower", "lower hold")
# What the hold of a carried node leaves of a move, as a share of it: the hold adds HOLD_FACTOR times the mean
# diagonal entry of the node's block of the tissue matrix (fem/node_holds.py), and the largest eigenvalue of that
# block is at most its trace, three times that mean.
HOLD_REMAINDER = 3.0 / HOLD_FACTOR


def _norm(vector):
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def _clean(value):
    """A value that JSON can hold: arrays as lists, nonfinite numbers as None."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _rotation_between(first, second):
    """Angle between two rotations given as quaternions (xyzw) [rad], accurate for small angles."""
    first, second = (np.asarray(value, dtype=float) / np.linalg.norm(value) for value in (first, second))
    # Vector and scalar part of conj(first) * second.
    vector = first[3] * second[:3] - second[3] * first[:3] - np.cross(first[:3], second[:3])
    return 2.0 * math.atan2(float(np.linalg.norm(vector)), abs(float(first @ second)))


def _largest(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


class Measurements:
    """Per-frame records of a run and their summary by phase.

    Args:
        scene: The ``ToolTissueScene``.
        path: The ``ToolPath`` that the driven tool follows.
    """

    def __init__(self, scene, path):
        self.scene, self.path = scene, path
        tool = scene.graspers[0]
        self.palm = tool.palm_body
        self.radius = scene.jaw_contact_radius
        self.records = []
        self.intervals = []
        self.failures = []
        # Joints of the driven tool: its arm and its two jaws.
        self.tool_joints = [*tool.arm_dof_indices, *tool.jaw_dof_indices]
        # Coordinates of the two jaw hinges; the second jaw opens with a negative angle.
        self.jaw_coordinates = list(tool.jaw_q_indices)
        self.skipped = []
        self.unused = []
        self._intervals = (0, 0, 0)
        # Counted intervals of the pinch (clamping, sticking, slipping, per tool) at the last frame record.
        self._pinch_counts = None
        # The pinch phase: host tables of the tissue (made at its first frame), the state at the end of the squeeze
        # hold that rises are measured from, and the numbers at the ends of its segments by their labels.
        self._pinch_tables = None
        self._pinch_reference = None
        # The carried nodes of the frame before (their places, forces and jaws): the work of a frame is taken
        # between two frames that follow each other.
        self._pinch_previous = None
        self.pinch_marks = {}
        # Planned compliance of the tool at the two holds of the pinch (ToolTissueScene.actuator_compliance), by label.
        self.pinch_compliance = {}
        # Centres of mass of the bodies in their frames (a host copy, made once).
        self._body_com = scene.model.body_com.numpy().astype(float)

    def add(self, frame, observation, *, step_s, period_s, observe_s, clock_s=None, started_at=None):
        """Record the frame whose end state ``observation`` holds; returns the record.

        ``clock_s`` is the time of ``time.perf_counter()`` at which the step of the frame began: the
        monotonic clock, on which a slow frame can be compared with the logs of the machine.
        ``started_at`` is the frame at whose end pose the target of this frame started, -1 for the
        start pose of the path; None is the frame before, as on a path that is run frame by frame.
        """
        path, scene = self.path, self.scene
        index = min(frame, path.frames - 1)
        segment = path.segment_of(index)
        target = path.poses[index]
        palm = np.asarray(observation["body_q"][self.palm], dtype=float)
        depths = observation["part_depth_m"]
        jaws = _largest([depths["PSM1/jaw_1"], depths["PSM1/jaw_2"]])
        contact = observation["contact"]
        counts = (
            observation["accepted_intervals"],
            observation["rejected_intervals"],
            observation["failed_intervals"],
        )
        accepted, rejected, failed = (now - before for now, before in zip(counts, self._intervals, strict=True))
        self._intervals = counts
        before = index - 1 if started_at is None else started_at
        previous = path.tip[before] if before >= 0 else path.start_pose[:3] + (path.tip[0] - path.poses[0, :3])
        commanded_travel = _norm(path.tip[index] - previous) / scene.args.substeps if frame < path.frames else 0.0
        crossing = observation["tip_crossing"] or {}
        record = {
            "frame": frame,
            "time_s": observation["time_s"],
            "clock_s": clock_s,
            "phase": segment.phase,
            "segment": segment.label,
            "hold": segment.hold,
            "commanded_depth_m": float(path.depth[index]),
            # The two jaw hinges and their command: how far each jaw is open [rad].
            "commanded_half_opening_rad": float(path.opening[index]),
            "jaw_half_opening_rad": [
                float(sign * observation["joint_q"][coordinate])
                for coordinate, sign in zip(self.jaw_coordinates, (1.0, -1.0), strict=True)
            ],
            "jaw_depth_m": jaws,
            "wrist_depth_m": depths["PSM1/wrist"],
            "shaft_depth_m": depths["PSM1/shaft"],
            "parked_tool_depth_m": _largest([value for name, value in depths.items() if name.startswith("PSM2/")]),
            "dimple_depth_m": observation["dimple_depth_m"],
            "surface_displacement_m": observation["surface_displacement_m"],
            **self._contact_numbers(observation),
            **self._pinch_counted(contact),
            **self._pinch_frame(frame, segment, observation),
            "tissue_force_N": _norm(contact["tissue_force_N"]),
            "tool_force_N": _norm(contact["tool_force_N"]),
            "tissue_force_vector_N": list(contact["tissue_force_N"]),
            "tool_force_vector_N": list(contact["tool_force_N"]),
            "loaded_samples": contact.get("loaded_samples", 0),
            "palm_position_error_m": _norm(palm[:3] - target[:3]),
            "palm_rotation_error_rad": _rotation_between(palm[3:], target[3:]),
            "tip_travel_per_interval_m": observation["tip_travel_m"],
            "commanded_tip_travel_per_interval_m": commanded_travel,
            # The tip sphere and the surface at rest: the gap after the frame, and for an interval of the frame
            # in which the gap changed sign from clear to inside, the travel of the tip in that interval and the
            # gap at its start and at its end.
            "tip_gap_m": observation["tip_gap_m"],
            "tip_crossing_travel_m": crossing.get("travel_m"),
            "tip_crossing_gap_before_m": crossing.get("gap_before_m"),
            "tip_crossing_gap_after_m": crossing.get("gap_after_m"),
            "accepted_intervals": accepted,
            "rejected_intervals": rejected,
            "failed_intervals": failed,
            "step_ms": 1e3 * step_s,
            "period_ms": None if period_s is None else 1e3 * period_s,
            "observe_ms": 1e3 * observe_s,
        }
        self.records.append(record)
        return record

    def _contact_numbers(self, observation):
        """What both a frame record and an interval record hold of the capsules, the contact and the actuators."""
        capsules = observation["capsule_depth_m"]
        jaws = [capsules["PSM1/jaw_1"], capsules["PSM1/jaw_2"]]
        driven = [capsules[name] for name in capsules if name.startswith("PSM1/")]
        contact = observation["contact"]
        tool_force = contact.get("tool_force_by_tool_N", [contact["tool_force_N"]])[0]
        demand = np.asarray(observation["actuator_demand_over_limit"])[self.tool_joints]
        return {
            # Exact drawn capsules of the driven tool, at the end of the interval.
            "capsule_yielding_depth_m": max(item["yielding"] for item in driven),
            "capsule_held_depth_m": max(item["held"] for item in driven),
            "capsule_centre_depth_m": max(item["centre"] for item in driven),
            "jaw_capsule_depth_m": max(max(item["yielding"], item["held"]) for item in jaws),
            "drawn_capsule_depth_m": max(item["drawn"] for item in driven),
            "drawn_centre_depth_m": max(item["drawn_centre"] for item in driven),
            **self._cad_numbers(observation),
            "tool_force_N": _norm(tool_force),
            "tool_force_vector_N": list(tool_force),
            "stored_force_N": contact.get("stored_force_N", 0.0),
            "held_nodes": contact.get("held_nodes", 0),
            "limited_nodes": contact.get("limited_nodes", 0),
            "touching_pieces": (contact.get("touching_pieces") or [0])[0],
            "inward_probes": contact.get("inward_probes", 0),
            "actuator_demand_over_limit": float(demand.max()),
            "actuator_demand_joint": int(self.tool_joints[int(demand.argmax())]),
            **self._pinch_numbers(contact),
        }

    @staticmethod
    def _pinch_numbers(contact):
        """What the pinch of the driven tool holds at the observed state (None without a pinch).

        Carried and clamped nodes of its two jaws, the contact's force on each jaw along its
        closing direction, squeeze, pull across the closing direction, the limit 2 mu S and
        the state of the pair.
        """
        pinch = contact.get("pinch")
        if pinch is None:
            return {"pinch_state": None}
        return {
            "pinch_state": pinch["state"][0],
            "pinch_carried_nodes": list(pinch["carried_nodes"][0]),
            "pinch_clamped_nodes": list(pinch["clamped_nodes"][0]),
            "pinch_jaw_normal_N": list(pinch["jaw_normal_N"][0]),
            "pinch_squeeze_N": pinch["squeeze_N"][0],
            "pinch_pull_N": pinch["pull_N"][0],
            "pinch_limit_N": pinch["limit_N"][0],
            # The two reports of the carry (running maxima over the sticking intervals since the last reset).
            "pinch_carry_residual_m": pinch["carry_residual_m"][0],
            "pinch_place_travel_m": pinch["place_travel_m"][0],
        }

    def _tables(self):
        """Host tables of the tissue block for the measurements of the pinch."""
        if self._pinch_tables is None:
            scene = self.scene
            rest = np.asarray(scene.selection.vertices, dtype=float)
            tetrahedra = np.asarray(scene.selection.indices).reshape(-1, 4)
            edges = np.stack([rest[tetrahedra[:, corner]] - rest[tetrahedra[:, 0]] for corner in (1, 2, 3)], axis=2)
            upper = _upper_surface_nodes(scene)
            site = np.asarray(self.path.derivation["pinch_site_m"], dtype=float)
            distance = np.linalg.norm(rest[upper, :2] - site, axis=1)
            self._pinch_tables = {
                "rest": rest,
                "tetrahedra": tetrahedra,
                "inverse": np.linalg.inv(edges),
                "upper": upper,
                "rings": [np.abs(distance - radius) < TENT_RING_HALF_WIDTH_M for radius in TENT_RINGS_M],
                "near": distance < TENT_RINGS_M[0],
            }
        return self._pinch_tables

    @staticmethod
    def _transform(pose, points):
        """Points of a body frame in the world, for a pose (xyz, xyzw)."""
        vector, scalar = np.asarray(pose[3:6], dtype=float), float(pose[6])
        return pose[:3] + points + 2.0 * np.cross(vector, np.cross(vector, points) + scalar * points)

    @staticmethod
    def _transform_inverse(pose, points):
        """World points in the frame of a body with the pose (xyz, xyzw)."""
        vector, scalar = -np.asarray(pose[3:6], dtype=float), float(pose[6])
        points = points - pose[:3]
        return points + 2.0 * np.cross(vector, np.cross(vector, points) + scalar * points)

    def _pinch_frame(self, frame, segment, observation):
        """What a frame of the pinch phase holds of the tissue between the jaws (nothing without a pinch).

        From the observation alone (the positions of the tissue nodes, the body poses and the
        per-node numbers of the pinch's diagnostics):

        - ``pinch_tissue_force_vector_N``: what the carried tissue gives the two jaws beside
          the contact's own reaction, summed from the elastic forces of the carried nodes by
          the rule of the pinch (the force on a node less what the contact returned of it;
          across the closing direction of its jaw to that jaw; along it to that jaw for a
          node that the contact holds, and half to each jaw for one that only the pinch
          holds). It is taken from the tissue, not from the wrench that the pinch applies.
        - the smallest volume ratio of a tetrahedron, in the block and at the carried nodes, and
          the smallest principal stretch of the tetrahedra without a carried node (the tent);
        - the work that the tool paid and the work that the tissue received through the
          carried nodes since the frame before (:meth:`_pinch_work`);
        - from the end of the squeeze hold on (the reference): the mean rise of the nodes
          that were carried then, the mean rise of the points of the jaw bodies that coincided
          with them (their places on the jaws), how far the node that is farthest behind its
          place is behind it, the rise of the palm and the rise of the upper surface on two
          rings around the pinch site.
        """
        contact = observation["contact"]
        pinch, q = contact.get("pinch"), observation.get("particle_q")
        if pinch is None or q is None or segment.phase != PINCH or frame >= self.path.frames:
            return {}
        tables = self._tables()
        tool = self.scene.graspers[0]
        q = np.asarray(q, dtype=float)
        body_q = np.asarray(observation["body_q"], dtype=float)
        ids = [np.asarray(values, dtype=np.int64) for values in pinch["carried_particles"][0]]
        pull = np.zeros(3)
        # The tissue's wrench on the bodies of the tool about their centres of mass: the law's, and on the jaws
        # what their carried nodes give them beside it.
        law_wrench = contact.get("body_wrench_N_Nm", {})
        wrenches = {
            int(body): np.asarray(law_wrench.get(int(body), np.zeros(6)), dtype=float).copy()
            for body in (tool.palm_body, *tool.jaw_bodies)
        }
        exchanged = []
        for jaw, body in enumerate(tool.jaw_bodies):
            force = np.asarray(pinch["carried_elastic_N"][0][jaw], dtype=float).reshape(-1, 3)
            law_part = np.asarray(pinch["carried_law_part_N"][0][jaw], dtype=float).reshape(-1, 3)
            rest_force = force - law_part
            by_law = np.asarray(pinch["carried_by_law"][0][jaw], dtype=bool)
            closing = np.asarray(pinch["jaw_closing_direction"][0][jaw], dtype=float)
            along = np.outer(rest_force @ closing, closing)
            # A node that only the pinch holds gives its part along the closing direction half to each jaw.
            half = np.where(by_law[:, None], 0.0, 0.5 * along)
            given = (rest_force - along) + np.where(by_law[:, None], along, half)
            # Per node: its elastic force and what its own jaw and the other jaw get of it (the law's part and
            # the pinch's).
            exchanged.append((force, law_part + given, half))
            pull += given.sum(axis=0) + half.sum(axis=0)
            for target, part in ((body, given), (tool.jaw_bodies[1 - jaw], half)):
                centre = self._transform(body_q[target], self._body_com[target])
                wrenches[int(target)][:3] += part.sum(axis=0)
                wrenches[int(target)][3:] += np.cross(q[ids[jaw]] - centre, part).sum(axis=0)
        x = q[tables["tetrahedra"]]
        deformation = np.stack([x[:, corner] - x[:, 0] for corner in (1, 2, 3)], axis=2) @ tables["inverse"]
        volume = np.linalg.det(deformation)
        carried = np.concatenate(ids)
        touched = np.isin(tables["tetrahedra"], carried).any(axis=1) if len(carried) else np.zeros(0, dtype=bool)
        # The tent: the tetrahedra without a node that is carried now or was at the end of the squeeze hold.
        gripped = (
            carried
            if self._pinch_reference is None
            else np.union1d(carried, np.concatenate(self._pinch_reference["ids"]))
        )
        tent = ~np.isin(tables["tetrahedra"], gripped).any(axis=1)
        stretch = np.linalg.svd(deformation[tent], compute_uv=False)[:, -1] * np.sign(volume[tent])
        law_force = contact.get("tool_force_by_tool_N", [contact["tool_force_N"]])[0]
        numbers = {
            "pinch_tissue_force_vector_N": pull.tolist(),
            "pinch_law_force_vector_N": list(law_force),
            "pinch_min_volume_ratio": float(volume.min()),
            "pinch_min_volume_ratio_carried": float(volume[touched].min()) if touched.any() else None,
            "pinch_tent_min_stretch": float(stretch.min()),
            "pinch_slip_length_m": pinch["slip_length_m"][0],
            "pinch_kept_nodes": pinch["kept_nodes"],
            "pinch_walking_nodes": pinch.get("walking_nodes", 0),
            "pinch_palm_height_m": float(body_q[self.palm, 2]),
            "pinch_palm_target_height_m": float(self.path.poses[frame][2]),
            # Largest distance from rest of an upper surface node within the first ring of the pinch site.
            "pinch_near_displacement_m": float(
                np.linalg.norm(q[tables["upper"]] - tables["rest"][tables["upper"]], axis=1)[tables["near"]].max()
            ),
        }
        numbers.update(
            self._pinch_work(
                {
                    "frame": frame,
                    "ids": carried,
                    "body": np.concatenate(
                        [np.full(len(mine), body) for mine, body in zip(ids, tool.jaw_bodies, strict=True)]
                    ).astype(np.int64),
                    "other": np.concatenate(
                        [np.full(len(mine), body) for mine, body in zip(ids, tool.jaw_bodies[::-1], strict=True)]
                    ).astype(np.int64),
                    "x": q[carried],
                    "force": np.concatenate([item[0] for item in exchanged]),
                    "tool": np.concatenate([item[1] for item in exchanged]),
                    "tool_other": np.concatenate([item[2] for item in exchanged]),
                    "body_q": body_q,
                }
            )
        )
        # Where each jaw holds its carried nodes: their distances from its hinge axis (z of the jaw body through
        # its origin) and their middle in the frame of the jaw.
        local = [
            self._transform_inverse(body_q[body], q[mine]) for body, mine in zip(tool.jaw_bodies, ids, strict=True)
        ]
        numbers["pinch_hinge_distance_m"] = [
            [
                float(value)
                for value in (np.linalg.norm(points[:, :2], axis=1).min(), np.linalg.norm(points[:, :2], axis=1).mean())
            ]
            if len(points)
            else None
            for points in local
        ]
        last = frame == segment.last_frame
        if last and segment.label == "squeeze hold":
            self._pinch_reference = {
                "frame": frame,
                "ids": ids,
                "heights": [q[mine, 2].copy() for mine in ids],
                "local": local,
                "upper": q[tables["upper"], 2].copy(),
                "palm": numbers["pinch_palm_height_m"],
                "palm_target": numbers["pinch_palm_target_height_m"],
                "slip": numbers["pinch_slip_length_m"],
            }
        reference = self._pinch_reference
        if reference is not None and sum(len(mine) for mine in reference["ids"]):
            node_rise = np.concatenate(
                [q[mine, 2] - heights for mine, heights in zip(reference["ids"], reference["heights"], strict=True)]
            )
            point_rise = np.concatenate(
                [
                    self._transform(body_q[body], points)[:, 2] - heights
                    for body, points, heights in zip(
                        tool.jaw_bodies, reference["local"], reference["heights"], strict=True
                    )
                ]
            )
            up = q[tables["upper"], 2] - reference["upper"]
            # Each carried node against its own place on its jaw: negative where the node is behind (below) it.
            node_lag = node_rise - point_rise
            numbers.update(
                pinch_carried_rise_m=float(node_rise.mean()),
                pinch_jaw_point_rise_m=float(point_rise.mean()),
                pinch_farthest_behind_m=float(max(0.0, -node_lag.min())),
                pinch_palm_rise_m=numbers["pinch_palm_height_m"] - reference["palm"],
                pinch_commanded_rise_m=numbers["pinch_palm_target_height_m"] - reference["palm_target"],
                pinch_ring_rise_m=[float(np.median(up[ring])) if ring.any() else None for ring in tables["rings"]],
                pinch_slip_since_reference_m=numbers["pinch_slip_length_m"] - reference["slip"],
            )
        if last:
            mark = {"frame": frame, **numbers}
            if reference is not None and sum(len(mine) for mine in reference["ids"]):
                mark["pinch_node_lag_m"] = node_lag.tolist()
            mark.update(
                {
                    name: record_value
                    for name, record_value in self._pinch_numbers(contact).items()
                    if name.startswith("pinch_")
                }
            )
            demand = np.asarray(observation["actuator_demand_over_limit"])[list(tool.jaw_dof_indices)]
            mark["jaw_actuator_torque_Nm"] = (demand * tool.config.jaw_torque_limit_nm).tolist()
            # The tissue's wrench on the palm and the jaws at this state, taken from the tissue: what the planned
            # compliance of the tool turns into a displacement of the palm.
            mark["tissue_wrench_by_body_N_Nm"] = {body: wrench.tolist() for body, wrench in wrenches.items()}
            self.pinch_marks[segment.label] = mark
        return numbers

    def _pinch_work(self, state):
        """Work through the carried nodes over the frame that ``state`` ends [J]; nothing for the first frame of a run.

        ``state`` holds the carried nodes at the end of a frame: their particles, their jaw
        bodies and the other jaw of each, places, elastic forces ``f``, what the node's own jaw
        gets of each force (``g``: the contact's part along its push direction and the pinch's
        part) and what the other jaw gets (``g'``: half of the part along the closing direction
        of a node that only the pinch holds), and the body poses. For the nodes that the same
        jaw carried at both ends of the frame, with ``M`` the motion of the node and ``U``,
        ``U'`` the motions of the points of its own and of the other jaw body that coincided
        with it at the start:

            pinch_work_tool_J    = - sum of 1/2 (g_start + g_end) . U + 1/2 (g'_start + g'_end) . U'
            pinch_work_tissue_J  = - sum of 1/2 (f_start + f_end) . M

        what the tool paid and what the tissue received. ``pinch_work_changed_nodes`` counts
        the nodes carried at one end only.
        """
        previous, self._pinch_previous = self._pinch_previous, state
        if previous is None or previous["frame"] != state["frame"] - 1:
            return {}
        _, before, after = np.intersect1d(previous["ids"], state["ids"], return_indices=True)
        same = previous["body"][before] == state["body"][after]
        before, after = before[same], after[same]
        x = previous["x"][before]
        moved = state["x"][after] - x

        def travel_of(name):
            """Travel over the frame of the points of the bodies ``name`` that coincided with the nodes at its start."""
            travel = np.zeros_like(x)
            for body in np.unique(previous[name][before]):
                rows = previous[name][before] == body
                local = self._transform_inverse(previous["body_q"][body], x[rows])
                travel[rows] = self._transform(state["body_q"][body], local) - x[rows]
            return travel

        def work(name, travel):
            return float(-0.5 * np.einsum("ij,ij->", previous[name][before] + state[name][after], travel))

        return {
            "pinch_work_tool_J": work("tool", travel_of("body")) + work("tool_other", travel_of("other")),
            "pinch_work_tissue_J": work("force", moved),
            "pinch_work_nodes": len(before),
            "pinch_work_changed_nodes": len(previous["ids"]) + len(state["ids"]) - 2 * len(before),
        }

    def pinch_rows(self, tolerance):
        """The rows of the self-check of the pinch phase, from the marks at the ends of its segments.

        Each row is a dictionary with its number, its name, whether it passed (None: the
        frames it needs were not run) and its numbers as a text. ``tolerance`` is the tracking
        tolerance of the tool [m] (``Example._tolerance``).

        1. Something is clamped: at the end of the squeeze hold each jaw has a clamped node and
           the pair sticks.
        2. The squeeze is the jaws': at the end of the squeeze hold and of the lift hold the
           contact's force on each jaw along its closing direction is positive and no larger
           than the torque limit of the jaw's actuator over the distance of its nearest
           carried node from the hinge.
        3. The tissue is at its places on the jaws, in two rows (:meth:`rise_rows`). 3a, at
           rest: at the end of the lift hold and of the lower hold every node carried at
           the end of the squeeze hold is as high above where it was then as the point of
           its jaw body that coincided with it (its place), within the tolerance, and the
           slip length has not grown. 3b, moving: in the intervals in which the pair stuck
           while the closed jaws moved together (from the end of the squeeze hold, where
           the example starts the two reports of the pinch anew, to the end of the last
           closed segment), no carried node was further from its place when an interval
           ended than that place travels in one interval, and at the end of the lift the
           tissue has risen with its places.
        4. The tool carries the pull: between the two holds the palm stays behind its commanded
           rise by what the planned compliance of the tool makes of the change of the tissue's
           wrench on its palm and jaws (the contact's wrench plus the forces of the carried
           nodes at those nodes, taken from their elastic forces and not from what the pinch
           applies), within the tolerance plus half of what the two planned configurations
           differ by for that change.
        5. A tent forms: at the end of the lift hold the rise of the upper surface falls from
           the carried nodes over the first ring to the second and is positive there.
        6. Nothing breaks: the smallest volume ratio of a tetrahedron is positive in every
           frame of the phase and no interval of the phase is rejected.
        7. Release: at the end of the hold after the release no node is carried, kept or
           walked by the pinch and every upper surface node within the first ring of the
           pinch site is within the release tolerance of rest.
        8. The work balances (:meth:`work_row`): from the end of the squeeze hold to the end
           of the lower hold the tool has paid through the carried nodes what the tissue
           has received through them, within the one-interval lag of the reaction.
        9. The grip holds the lift (:meth:`grip_row`): the pair has not slipped while the
           jaws were closed, the friction limit ``2 mu S`` was at least the pull in every
           frame of that time, and at the end of the lift hold the tissue of the tent is
           above the stretch limit of the material.
        """
        marks = self.pinch_marks
        squeeze, lifted, lift_hold = marks.get("squeeze hold"), marks.get("lift"), marks.get("lift hold")
        lowered, release = marks.get("lower hold"), marks.get("release hold")
        limit = self.scene.graspers[0].config.jaw_torque_limit_nm
        rows = []

        def row(number, name, passed, text):
            rows.append({"row": number, "name": name, "passed": passed, "numbers": text})

        if squeeze is None:
            row(1, "something is clamped", None, "the squeeze hold was not run")
        else:
            carried, clamped = squeeze["pinch_carried_nodes"], squeeze["pinch_clamped_nodes"]
            row(
                1,
                "something is clamped",
                bool(min(clamped) > 0 and squeeze["pinch_state"] == "stick"),
                f"end of the squeeze hold: {squeeze['pinch_state']}, carried {carried[0]} + {carried[1]}, clamped "
                f"{clamped[0]} + {clamped[1]}",
            )
        holds = [(label, mark) for label, mark in (("squeeze hold", squeeze), ("lift hold", lift_hold)) if mark]
        if len(holds) < 2:
            row(2, "the squeeze is the jaws'", None, "the two holds were not run")
        else:
            passed, texts = True, []
            for label, mark in holds:
                for jaw in (0, 1):
                    force, reach = mark["pinch_jaw_normal_N"][jaw], mark["pinch_hinge_distance_m"][jaw]
                    if reach is None:
                        passed = False
                        texts.append(f"{label}, jaw {jaw + 1}: no carried node")
                        continue
                    bound = limit / reach[0]
                    torque = abs(mark["jaw_actuator_torque_Nm"][jaw])
                    share = force * reach[1] / torque if torque > 0 else float("inf")
                    passed = passed and 0.0 < force <= bound
                    texts.append(
                        f"{label}, jaw {jaw + 1}: {1e3 * force:.1f} mN of at most {1e3 * bound:.0f} mN (torque limit "
                        f"over {1e3 * reach[0]:.2f} mm), {share:.2f} of what its actuator presses"
                    )
            row(2, "the squeeze is the jaws'", bool(passed), "; ".join(texts))
        if squeeze is None or lifted is None or lift_hold is None or "pinch_carried_rise_m" not in lift_hold:
            row("3a", "the tissue is at its places on the jaws at rest", None, "the lift and its hold were not run")
            row("3b", "the tissue stays at its places on moving jaws", None, "the lift was not run")
        else:
            # The reports of the carry over the closed segments: the mark of the last of them that was run.
            last = next(marks[label] for label in reversed(PINCH_CLOSED_SEGMENTS) if label in marks)
            for number, name, passed, text in self.rise_rows(lifted, lift_hold, lowered, last, tolerance):
                if number == "3b":
                    # While the jaws closed the contact placed the nodes on the faces: reported, not judged.
                    text += (
                        f"; while the jaws closed, where the contact places the nodes on the faces: "
                        f"{1e6 * squeeze['pinch_carry_residual_m']:.2f} um against a travel of "
                        f"{1e6 * squeeze['pinch_place_travel_m']:.2f} um"
                    )
                row(number, name, passed, text)
        planned = [self.pinch_compliance.get(label) for label in ("squeeze hold", "lift hold")]
        if squeeze is None or lift_hold is None or "pinch_palm_rise_m" not in lift_hold or None in planned:
            row(4, "the tool carries the pull", None, "the two holds were not run, or no compliance was planned")
        else:
            change = {
                body: np.asarray(lift_hold["tissue_wrench_by_body_N_Nm"][body])
                - np.asarray(squeeze["tissue_wrench_by_body_N_Nm"][body])
                for body in squeeze["tissue_wrench_by_body_N_Nm"]
            }
            # Displacement of the palm along the lift that the change of the wrench gives, by either plan.
            gives = [float(plan(change)[2]) for plan in planned]
            sag = lift_hold["pinch_commanded_rise_m"] - lift_hold["pinch_palm_rise_m"]
            predicted = -0.5 * (gives[0] + gives[1])
            allowed = tolerance + 0.5 * abs(gives[1] - gives[0])
            force = sum(wrench[2] for wrench in change.values())
            pinch_part = lift_hold["pinch_tissue_force_vector_N"][2] - squeeze["pinch_tissue_force_vector_N"][2]
            row(
                4,
                "the tool carries the pull",
                bool(abs(sag - predicted) <= allowed),
                f"palm {1e3 * lift_hold['pinch_palm_rise_m']:.3f} mm up of {1e3 * lift_hold['pinch_commanded_rise_m']:.3f} "
                f"commanded: sag {1e3 * sag:.3f} mm; the tissue's force on the tool along the lift changed by "
                f"{1e3 * force:.1f} mN ({1e3 * pinch_part:.1f} mN of it through the carried nodes); with the planned "
                f"compliance of the tool at the two holds that wrench gives a sag of {-1e3 * gives[0]:.3f} and "
                f"{-1e3 * gives[1]:.3f} mm: predicted {1e3 * predicted:.3f} mm, difference "
                f"{1e6 * (sag - predicted):+.1f} um, allowed {1e6 * allowed:.1f}",
            )
        if lift_hold is None or "pinch_ring_rise_m" not in lift_hold:
            row(5, "a tent forms", None, "the lift hold was not run")
        else:
            rise, rings = lift_hold["pinch_carried_rise_m"], lift_hold["pinch_ring_rise_m"]
            passed = None not in rings and rise > rings[0] > rings[1] > 0.0
            # A flat punch on a half-space lifts the surface at the distance r by (2 / pi) asin(a / r) of its own
            # rise; a: radius of a disc of the area of the carried patch, taken as the box of the carried nodes
            # seen from above at rest.
            tables = self._tables()
            nodes = np.concatenate(self._pinch_reference["ids"])
            box = np.ptp(tables["rest"][nodes, :2], axis=0) if len(nodes) else np.zeros(2)
            radius = math.sqrt(float(box[0] * box[1]) / math.pi)
            punch = [(2.0 / math.pi) * math.asin(min(1.0, radius / ring)) * rise for ring in TENT_RINGS_M]
            row(
                5,
                "a tent forms",
                bool(passed),
                f"end of the lift hold: carried nodes {1e3 * rise:.3f} mm up, upper surface "
                + ", ".join(
                    f"{1e3 * (value or 0.0):.3f} mm at {1e3 * ring:.0f} mm"
                    for value, ring in zip(rings, TENT_RINGS_M, strict=True)
                )
                + f" from the pinch (a flat punch of {1e3 * radius:.2f} mm radius on a half-space: "
                + ", ".join(f"{1e3 * value:.3f} mm" for value in punch)
                + ")",
            )
        records = [record for record in self.records if record["phase"] == PINCH and "pinch_min_volume_ratio" in record]
        if not records:
            row(6, "nothing breaks", None, "the phase was not run")
        else:
            smallest = min(records, key=lambda record: record["pinch_min_volume_ratio"])
            carried = [r for r in records if r["pinch_min_volume_ratio_carried"] is not None]
            least = min(carried, key=lambda record: record["pinch_min_volume_ratio_carried"]) if carried else None
            failures = [failure for failure in self.failures if failure["phase"] == PINCH]
            rejected = sum(record["rejected_intervals"] + record["failed_intervals"] for record in records)
            rejected += sum(failure["rejected_intervals"] for failure in failures)
            row(
                6,
                "nothing breaks",
                bool(smallest["pinch_min_volume_ratio"] > 0.0 and rejected == 0 and not failures),
                f"smallest volume ratio {smallest['pinch_min_volume_ratio']:.3f} (frame {smallest['frame']}, "
                f"{smallest['segment']})"
                + (
                    f", at carried nodes {least['pinch_min_volume_ratio_carried']:.3f} (frame {least['frame']}, "
                    f"{least['segment']})"
                    if least
                    else ""
                )
                + f"; {rejected} rejected or failed intervals in {len(records)} frames",
            )
        if release is None:
            row(7, "release", None, "the hold after the release was not run")
        else:
            carried = sum(release["pinch_carried_nodes"])
            kept, walking = release["pinch_kept_nodes"], release["pinch_walking_nodes"]
            away = release["pinch_near_displacement_m"]
            opened = [r for r in records if r["segment"] in ("release", "release hold")]
            busy = [r for r in opened if r["pinch_kept_nodes"] or r["pinch_walking_nodes"]]
            end = next((r["time_s"] for r in opened if r["segment"] == "release hold"), None)
            if busy and end is not None:
                since = busy[-1]["time_s"] + self.scene.frame_dt - (end - self.scene.frame_dt)
                last = f"the last node kept or walked by the pinch {since:+.2f} s after the jaws reached their command"
            else:
                last = "no node kept or walked by the pinch while the jaws opened"
            row(
                7,
                "release",
                bool(carried == 0 and kept == 0 and walking == 0 and away <= RELEASE_TOLERANCE_M),
                f"end of the release hold: {carried} carried, {kept} kept, {walking} walked; upper surface within "
                f"{1e3 * TENT_RINGS_M[0]:.0f} mm of the pinch {1e6 * away:.1f} um from rest at most (allowed "
                f"{1e6 * RELEASE_TOLERANCE_M:.0f}); {last}",
            )
        slips = self.pinch_slips()
        row(*self.work_row(records, squeeze, lift_hold, lowered))
        row(*self.grip_row(records, squeeze, lifted, lift_hold, slips, self.scene.solver.min_stretch))
        return rows

    @staticmethod
    def rise_rows(lifted, lift_hold, lower_hold, last, tolerance):
        """Rows 3a and 3b of the self-check of the pinch: the carried tissue against its places on the jaws.

        ``lifted``, ``lift_hold`` and ``lower_hold`` are the marks at the end of the lift, of
        its hold and of the hold after the lowering (``None``: not run): the mean rise of the
        nodes that were carried at the end of the squeeze hold (``pinch_carried_rise_m``), the
        mean rise of the points of the jaw bodies that coincided with them then
        (``pinch_jaw_point_rise_m``), each node against its own point (``pinch_node_lag_m``)
        and the slip length since then. ``last`` is the mark of the last closed segment that
        was run: it holds the two reports of the pinch over every interval in which the pair
        stuck since the end of the squeeze hold (``pinch_carry_residual_m``,
        ``pinch_place_travel_m``; the example starts them anew there).

        3a. At rest, at the end of the lift hold and of the lower hold: every node is within
            ``tolerance`` of its place (the tracking tolerance of the tool, with which a
            resting tool is compared with its target) and the pair has not slipped. A hold
            that was not run leaves the row open unless the other one fails.
        3b. Moving: the largest distance of a carried node from its place at the end of an
            interval is at most ``(1 + HOLD_REMAINDER)`` times the largest travel of a place
            in one interval, and at the end of the lift the mean rise of the tissue is that
            of its places within ``tolerance`` plus that travel; the pair has not slipped.
            The row is for jaws that move together: while they close or open the contact
            itself moves the nodes on the faces, across the closing direction too, and the
            distance holds that move (it is printed for the closing, not judged).
            Where it comes from: a carried node is given the velocity that takes it to its
            place at the pose that the contact predicts for its jaw, a step at constant
            velocity. What is left when the interval has ended is the error of that
            prediction, the change of the travel from one interval to the next (at most one
            interval's travel while the jaws do not turn back within an interval), and what
            the hold of the node leaves of the move. A carry that is slow by a share ``s`` of
            the travel adds that share in every interval and is outside after ``1 / s``
            intervals.

        Returns:
            Two tuples (number, name, passed, text).
        """

        def compared(mark):
            lag = mark["pinch_carried_rise_m"] - mark["pinch_jaw_point_rise_m"]
            nodes = np.asarray(mark.get("pinch_node_lag_m") or [], dtype=float)
            away = int((np.abs(nodes) > tolerance).sum())
            text = (
                f"carried nodes {1e3 * mark['pinch_carried_rise_m']:.3f} mm up, their places on the jaws "
                f"{1e3 * mark['pinch_jaw_point_rise_m']:.3f} mm: {1e6 * lag:+.1f} um in the mean; {away} of "
                f"{len(nodes)} nodes more than {1e6 * tolerance:.1f} um from their place"
                + (f", the farthest {1e6 * nodes[np.abs(nodes).argmax()]:+.1f} um" if len(nodes) else "")
            )
            return lag, away, mark["pinch_slip_since_reference_m"], text

        passed, texts = True, []
        for label, mark in (("lift hold", lift_hold), ("lower hold", lower_hold)):
            if mark is None or "pinch_node_lag_m" not in mark:
                passed = None if passed else passed
                texts.append(f"the {label} was not run")
                continue
            _, away, slipped, text = compared(mark)
            if away or slipped > 0.0:
                passed = False
            texts.append(f"end of the {label}: {text}; slipped {1e3 * slipped:.3f} mm since the squeeze hold")
        rest = ("3a", "the tissue is at its places on the jaws at rest", passed, "; ".join(texts))
        lag, _, slipped, text = compared(lifted)
        residual, travel = last["pinch_carry_residual_m"], last["pinch_place_travel_m"]
        allowed = (1.0 + HOLD_REMAINDER) * travel
        moving = (
            "3b",
            "the tissue stays at its places on moving jaws",
            bool(travel > 0.0 and residual <= allowed and abs(lag) <= tolerance + travel and slipped <= 0.0),
            f"in the intervals in which the pair stuck since the squeeze hold (up to frame {last['frame']}) a carried "
            f"node was at most "
            f"{1e6 * residual:.2f} um from its place on its jaw when an interval ended, across the closing direction; "
            f"allowed {1e6 * allowed:.2f} um: the largest travel of a place in one interval ({1e6 * travel:.2f} um) "
            f"and {HOLD_REMAINDER:.5f} of it for the hold of a node; end of the lift, the jaws have just moved: {text}, "
            f"allowed {1e6 * (tolerance + travel):.1f} um in the mean; slipped {1e3 * slipped:.3f} mm since the "
            "squeeze hold",
        )
        return rest, moving

    @staticmethod
    def work_row(records, squeeze, lift_hold, lowered):
        """Row 8 of the self-check of the pinch: what the tool pays through the carried nodes, the tissue receives.

        ``records`` are the frame records of the pinch phase with the work of each frame
        (:meth:`_pinch_work`), ``squeeze``, ``lift_hold`` and ``lowered`` the marks at the end
        of the squeeze hold, of the lift hold and of the lower hold. Over the frames after the
        squeeze hold up to the end of the lower hold:

            W_tool - W_tissue  =  sum of pinch_work_tool_J  -  sum of pinch_work_tissue_J

        is within ``T_max d_max + 2 T_max min(r_max, (1 + HOLD_REMAINDER) d_max)`` with ``T_max``
        the largest pull of those frames, ``d_max`` the largest travel of a place in one
        interval and ``r_max`` the largest distance of a carried node from its place at the
        end of an interval (the two reports of the pinch at the lower hold). The row passes
        when each stroke (the lift with its hold, the lowering with its hold) is within its
        half of that bound: a reaction that is short by a share pays too little on the way
        up and returns too little on the way down, which cancels over the cycle and not in a
        stroke. Where the bound comes from:

        - The reaction of a carried node reaches the tool one interval late: through an
          interval the tool feels the force of the state before it, the tissue is moved
          against a force that changes through it. The difference is half of the change of
          the force times the travel of the interval; over a lift the changes add up to the
          pull and the tool pays ``T d / 2`` less than the tissue receives, over the lowering
          it gets ``T d / 2`` more back than the tissue gives: ``T_max d_max`` for both.
        - A node that is ``r`` from its place is taken there against its force by the carry:
          work ``T r`` that the tissue receives and the tool has not paid along its own
          travel, once on the way up and once on the way down. ``r`` counts as far as a
          carry that keeps up can leave it (row 3b); a larger one does not widen the bound.

        The work of the frames is taken with the forces at the two ends of each frame, so
        the first term is not in the measured difference: the bound leaves room for it.
        Not in the balance, and printed: the work while the jaws close and open (placements
        of the contact, slip and release) and the contact's own force on the tool along the
        lift at the two holds (its stored stand-off).
        A pair that slipped is not judged: friction takes the difference.

        Returns:
            A tuple (number, name, passed, text).
        """
        name = "the work through the carried nodes balances"
        if squeeze is None or lowered is None:
            return 8, name, None, "the phase was not run from the squeeze hold to the lower hold"

        def total(key, frames):
            return sum(record.get(key, 0.0) for record in frames)

        closed = [record for record in records if squeeze["frame"] < record["frame"] <= lowered["frame"]]
        before = [record for record in records if record["frame"] <= squeeze["frame"]]
        after = [record for record in records if record["frame"] > lowered["frame"]]
        tool, tissue = total("pinch_work_tool_J", closed), total("pinch_work_tissue_J", closed)
        changed = total("pinch_work_changed_nodes", closed)
        missing = sum(1 for record in closed if "pinch_work_tool_J" not in record)
        pull = max(record["pinch_pull_N"] for record in [*closed, squeeze])
        travel, residual = lowered["pinch_place_travel_m"], lowered["pinch_carry_residual_m"]
        bound = pull * travel + 2.0 * pull * min(residual, (1.0 + HOLD_REMAINDER) * travel)
        slipped = lowered["pinch_slip_since_reference_m"]
        parts, inside = [], True
        for labels in (("lift", "lift hold"), ("lower", "lower hold")):
            frames = [record for record in closed if record["segment"] in labels]
            paid, received = total("pinch_work_tool_J", frames), total("pinch_work_tissue_J", frames)
            inside = inside and abs(paid - received) <= 0.5 * bound
            parts.append(f"{' and '.join(labels)}: tool {1e3 * paid:+.4f} mJ, tissue {1e3 * received:+.4f} mJ")
        text = (
            f"end of the squeeze hold to end of the lower hold: the tool paid {1e3 * tool:+.4f} mJ, the tissue received "
            f"{1e3 * tissue:+.4f} mJ: difference {1e3 * (tool - tissue):+.4f} mJ, allowed {1e3 * bound:.4f} mJ and half of "
            f"it in each stroke (largest "
            f"pull {1e3 * pull:.1f} mN, largest travel of a place in one interval {1e6 * travel:.2f} um, largest distance "
            f"of a node from its place {1e6 * residual:.2f} um); {'; '.join(parts)}; not in the balance: closing "
            f"{1e3 * total('pinch_work_tool_J', before):+.4f} / "
            f"{1e3 * total('pinch_work_tissue_J', before):+.4f} mJ and opening "
            f"{1e3 * total('pinch_work_tool_J', after):+.4f} / {1e3 * total('pinch_work_tissue_J', after):+.4f} mJ "
            "(tool / tissue), the contact's own force on the tool along the lift "
            f"{1e3 * squeeze['pinch_law_force_vector_N'][2]:+.1f} mN at the squeeze hold"
            + (f" and {1e3 * lift_hold['pinch_law_force_vector_N'][2]:+.1f} mN at the lift hold" if lift_hold else "")
            + (f"; {int(changed)} nodes carried at one end of a frame only" if changed else "")
            + (f"; {missing} frames without a frame before them" if missing else "")
        )
        if slipped > 0.0:
            return (
                8,
                name,
                None,
                f"not judged, the pair slipped {1e3 * slipped:.3f} mm (friction takes the difference); {text}",
            )
        return 8, name, bool(not missing and inside and abs(tool - tissue) <= bound), text

    @staticmethod
    def grip_row(records, squeeze, lifted, lift_hold, slips, stretch_limit):
        """Row 9 of the self-check of the pinch: the grip holds the lift.

        Passes when the pair has not slipped in a frame in which the jaws were commanded
        closed (``slips``, :meth:`pinch_slips`), the friction limit ``2 mu S`` was at least
        the pull at the end of every such frame, and at the end of the lift hold the smallest
        principal stretch of the tetrahedra without a carried node (the tent) is above
        ``stretch_limit``, the stretch limit of the material. Printed with it, as numbers and
        not as a second verdict: the squeeze at the end of the squeeze hold ``S_0``, the pull
        per rise of the carried tissue at the end of the lift ``k_T``, the rise at which that
        pull reaches the friction limit of ``S_0`` (``2 mu S_0 / k_T``: where the grip would
        end if the squeeze stayed as it is at the squeeze hold; it grows with the lift), and
        the limit over the pull at the end of the lift and of its hold.

        Returns:
            A tuple (number, name, passed, text).
        """
        name = "the grip holds the lift"
        if squeeze is None or lifted is None or lift_hold is None or "pinch_carried_rise_m" not in lifted:
            return 9, name, None, "the lift and its hold were not run"
        closed = [
            record
            for record in records
            if record["segment"] in PINCH_CLOSED_SEGMENTS and record["frame"] >= squeeze["frame"]
        ]
        short = [record for record in closed if record["pinch_limit_N"] < record["pinch_pull_N"]]
        stretch = lift_hold["pinch_tent_min_stretch"]
        slipped = slips["closed_intervals"] if slips else 0
        passed = slipped == 0 and not short and stretch > stretch_limit
        rise, pull = lifted["pinch_carried_rise_m"], lifted["pinch_pull_N"]
        stiffness = pull / rise if rise > 0.0 else float("inf")
        limit = squeeze["pinch_limit_N"]
        margins = [
            mark["pinch_limit_N"] / mark["pinch_pull_N"] if mark["pinch_pull_N"] > 0.0 else float("inf")
            for mark in (lifted, lift_hold)
        ]
        text = (
            (
                f"no slip while the jaws were closed, the friction limit at least the pull in all {len(closed)} frames"
                if slipped == 0 and not short
                else f"{slipped} slipping intervals while the jaws were closed, the friction limit below the pull at "
                f"the end of {len(short)} of {len(closed)} frames"
                + (f" (first: frame {short[0]['frame']}, {short[0]['segment']})" if short else "")
            )
            + f"; squeeze at the end of the squeeze hold S_0 = {1e3 * squeeze['pinch_squeeze_N']:.1f} mN, friction limit "
            f"2 mu S_0 = {1e3 * limit:.1f} mN; the tissue pulls with k_T = {stiffness:.1f} N/m of its rise (end of the "
            f"lift: {1e3 * pull:.1f} mN at {1e3 * rise:.3f} mm), so 2 mu S_0 / k_T = {1e3 * limit / stiffness:.2f} mm of "
            f"rise; limit over pull {margins[0]:.2f} at the end of the lift and {margins[1]:.2f} at the end of its hold "
            f"(squeeze {1e3 * lifted['pinch_squeeze_N']:.1f} and {1e3 * lift_hold['pinch_squeeze_N']:.1f} mN); smallest "
            f"stretch of the tent {stretch:.3f} at the end of the lift hold, limit of the material {stretch_limit:g}"
        )
        return 9, name, bool(passed), text

    def pinch_slips(self):
        """Slips of the pinch: where the pair slipped while the jaws were commanded closed and still, and elsewhere.

        Returns:
            None without a pinch record. Otherwise ``closed``: the frames of the segments in
            which the jaws are commanded closed and stand still in which the pair slipped,
            with the first of them (frame, segment, commanded rise, pull, limit) and the slip
            length at the last; ``moving``: the slipping intervals while the jaws close or
            open; ``slip_length_m``: the slip length at the end of the phase.
        """
        records = [record for record in self.records if record["phase"] == PINCH and "pinch_slip_length_m" in record]
        if not records:
            return None
        slipped = [record for record in records if record["pinch_slipping_intervals"][0] > 0]
        closed = [record for record in slipped if record["segment"] in PINCH_CLOSED_SEGMENTS]
        moving = [record for record in slipped if record["segment"] not in PINCH_CLOSED_SEGMENTS]
        first = closed[0] if closed else None
        return {
            "closed_frames": len(closed),
            "closed_intervals": sum(record["pinch_slipping_intervals"][0] for record in closed),
            "first": None
            if first is None
            else {
                "frame": first["frame"],
                "segment": first["segment"],
                "commanded_rise_m": first.get("pinch_commanded_rise_m"),
                "carried_rise_m": first.get("pinch_carried_rise_m"),
                "pull_N": first["pinch_pull_N"],
                "limit_N": first["pinch_limit_N"],
            },
            "last_frame": closed[-1]["frame"] if closed else None,
            "moving_intervals": sum(record["pinch_slipping_intervals"][0] for record in moving),
            "slip_length_m": records[-1]["pinch_slip_length_m"],
        }

    def _pinch_counted(self, contact):
        """Intervals of the frame in which a pair of jaws clamped, stuck and slipped, per tool (a frame record)."""
        pinch = contact.get("pinch")
        if pinch is None:
            return {}
        names = ("clamping_intervals", "sticking_intervals", "slipping_intervals")
        counts = [list(pinch[name]) for name in names]
        before = self._pinch_counts or [[0] * len(values) for values in counts]
        self._pinch_counts = counts
        return {
            f"pinch_{name}": [now - last for now, last in zip(values, earlier, strict=True)]
            for name, values, earlier in zip(names, counts, before, strict=True)
        }

    @staticmethod
    def _cad_numbers(observation):
        """How far the drawn parts of the driven tool (their CAD vertices) are inside the tissue as drawn [m]."""
        parts = {name: item for name, item in observation["drawn_part_depth_m"].items() if name.startswith("PSM1/")}
        jaws = [item for name, item in parts.items() if name.rsplit("/", 1)[1].startswith("jaw")]
        return {
            "cad_jaw_depth_m": max(max(item["yielding"], item["held"]) for item in jaws),
            "cad_jaw_yielding_depth_m": max(item["yielding"] for item in jaws),
            "cad_tool_depth_m": max(max(item["yielding"], item["held"]) for item in parts.values()),
            "cad_part_depth_m": {name: max(item["yielding"], item["held"]) for name, item in parts.items()},
        }

    def add_interval(self, frame, observation, dt):
        """Record the end of one accepted physics interval of ``frame`` (a measurement that is asked for)."""
        scene = self.scene
        body_q = np.asarray(observation["body_q"], dtype=float)
        pose = body_q[scene.tip_body]
        # The jaw tip in the world: for the speed of the tool between two intervals.
        rotation = pose[3:]
        local = np.asarray(scene.tip_local, dtype=float)
        vector = rotation[:3]
        tip = pose[:3] + local + 2.0 * np.cross(vector, np.cross(vector, local) + rotation[3] * local)
        index = min(frame, self.path.frames - 1)
        segment = self.path.segment_of(index)
        record = {
            "frame": frame,
            "time_s": observation["time_s"],
            "dt_s": dt,
            "phase": segment.phase,
            "segment": segment.label,
            "hold": segment.hold,
            "tip_m": tip.tolist(),
            "dimple_depth_m": observation["dimple_depth_m"],
            "rejected_intervals": observation["rejected_intervals"],
            **self._contact_numbers(observation),
        }
        self.intervals.append(record)
        return record

    def add_failure(self, frame, reason, restart):
        """Note that the scene could not go on in ``frame``; the frames up to ``restart`` are skipped."""
        segment = self.path.segment_of(min(frame, self.path.frames - 1))
        last = self.records[-1] if self.records else {}
        # The intervals of the frame that failed belong to its phase, not to the next recorded frame.
        scene = self.scene
        counts = (scene.accepted_intervals, scene.rejected_intervals, scene.failed_intervals)
        accepted, rejected, _ = (now - before for now, before in zip(counts, self._intervals, strict=True))
        self._intervals = counts
        # After a failure the contact is cleared, and the counted intervals of the pinch with it.
        self._pinch_counts = None
        self.failures.append(
            {
                "frame": frame,
                "phase": segment.phase,
                "segment": segment.label,
                "reason": reason,
                "accepted_intervals": accepted,
                "rejected_intervals": rejected,
                "commanded_depth_m": float(self.path.depth[min(frame, self.path.frames - 1)]),
                "last_jaw_depth_m": last.get("jaw_depth_m"),
                "last_dimple_depth_m": last.get("dimple_depth_m"),
                "last_tissue_force_N": last.get("tissue_force_N"),
                "restart_frame": restart,
            }
        )
        end = self.path.frames if restart is None else restart
        self.skipped.append((frame, end))

    def add_unused(self, first, end):
        """Note that the frames ``first`` to ``end - 1`` are not run: the rule of their segment ended it before them."""
        if end > first:
            self.unused.append((first, end))

    def phases(self):
        """One summary per phase: what was reached, the largest values, the hold ends, the frame times."""
        result = {}
        for number, phase in enumerate(PHASES, 1):
            first, last = self.path.phase_frames(phase)
            records = [record for record in self.records if record["phase"] == phase and record["frame"] <= last]
            failures = [failure for failure in self.failures if failure["phase"] == phase]
            skipped, unused = (
                sum(max(0, min(end, last + 1) - max(start, first)) for start, end in spans if end > first)
                for spans in (self.skipped, self.unused)
            )
            status = "not run"
            if records:
                status = (
                    "failed"
                    if failures
                    else ("completed" if len(records) + skipped + unused >= last - first + 1 else "partly run")
                )
            elif skipped:
                status = "skipped after a failure"
            summary = {
                "number": number,
                "status": status,
                "frames": len(records),
                "planned_frames": last - first + 1,
                "skipped_frames": skipped,
                # Frames that the rule of their segment made unnecessary (the fast approach ends once the tool
                # has crossed the surface).
                "unused_frames": unused,
                "failures": failures,
            }
            if records:
                steps = np.array([record["step_ms"] for record in records])
                periods = np.array([record["period_ms"] for record in records if record["period_ms"] is not None])
                holds = []
                for segment in self.path.segments:
                    if segment.phase == phase and segment.hold:
                        ends = [record for record in records if record["frame"] == segment.last_frame]
                        if ends:
                            end = ends[0]
                            holds.append(
                                {
                                    "segment": segment.label,
                                    "frame": segment.last_frame,
                                    "commanded_depth_m": end["commanded_depth_m"],
                                    "jaw_depth_m": end["jaw_depth_m"],
                                    "jaw_capsule_depth_m": end["jaw_capsule_depth_m"],
                                    "cad_jaw_depth_m": end["cad_jaw_depth_m"],
                                    "dimple_depth_m": end["dimple_depth_m"],
                                    "tissue_force_N": end["tissue_force_N"],
                                    "tool_force_N": end["tool_force_N"],
                                    "palm_position_error_m": end["palm_position_error_m"],
                                    "palm_rotation_error_rad": end["palm_rotation_error_rad"],
                                    "commanded_half_opening_rad": end["commanded_half_opening_rad"],
                                    "jaw_half_opening_rad": end["jaw_half_opening_rad"],
                                    "held_nodes": end["held_nodes"],
                                    **{name: value for name, value in end.items() if name.startswith("pinch_")},
                                }
                            )
                summary.update(
                    seconds=records[-1]["time_s"] - records[0]["time_s"] + self.scene.frame_dt,
                    commanded_depth_max_m=max(record["commanded_depth_m"] for record in records),
                    jaw_depth_max_m=_largest([record["jaw_depth_m"] for record in records]),
                    capsule_yielding_depth_max_m=max(record["capsule_yielding_depth_m"] for record in records),
                    capsule_held_depth_max_m=max(record["capsule_held_depth_m"] for record in records),
                    capsule_centre_depth_max_m=max(record["capsule_centre_depth_m"] for record in records),
                    drawn_capsule_depth_max_m=max(record["drawn_capsule_depth_m"] for record in records),
                    cad_jaw_depth_max_m=max(record["cad_jaw_depth_m"] for record in records),
                    cad_jaw_yielding_depth_max_m=max(record["cad_jaw_yielding_depth_m"] for record in records),
                    cad_tool_depth_max_m=max(record["cad_tool_depth_m"] for record in records),
                    cad_part_depth_max_m={
                        name: max(record["cad_part_depth_m"][name] for record in records)
                        for name in records[0]["cad_part_depth_m"]
                    },
                    actuator_demand_over_limit_max=max(record["actuator_demand_over_limit"] for record in records),
                    wrist_depth_max_m=_largest([record["wrist_depth_m"] for record in records]),
                    shaft_depth_max_m=_largest([record["shaft_depth_m"] for record in records]),
                    dimple_depth_max_m=max(record["dimple_depth_m"] for record in records),
                    surface_displacement_max_m=max(record["surface_displacement_m"] for record in records),
                    tissue_force_max_N=max(record["tissue_force_N"] for record in records),
                    tool_force_max_N=max(record["tool_force_N"] for record in records),
                    palm_position_error_max_m=max(record["palm_position_error_m"] for record in records),
                    tip_travel_per_interval_max_m=max(record["tip_travel_per_interval_m"] for record in records),
                    commanded_tip_travel_per_interval_max_m=max(
                        record["commanded_tip_travel_per_interval_m"] for record in records
                    ),
                    rejected_intervals=sum(record["rejected_intervals"] for record in records)
                    + sum(failure["rejected_intervals"] for failure in failures),
                    failed_intervals=len(failures),
                    step_ms_median=float(np.median(steps)),
                    step_ms_max=float(steps.max()),
                    period_ms_median=float(np.median(periods)) if len(periods) else None,
                    period_ms_max=float(periods.max()) if len(periods) else None,
                    holds=holds,
                )
                if "pinch_clamping_intervals" in records[0]:
                    # Intervals of the phase in which a pair of jaws clamped tissue, stuck and slipped, per tool,
                    # and the frames in which one clamped.
                    for name in ("clamping", "sticking", "slipping"):
                        summary[f"pinch_{name}_intervals"] = np.sum(
                            [record[f"pinch_{name}_intervals"] for record in records], axis=0
                        ).tolist()
                    summary["pinch_clamping_frames"] = [
                        record["frame"] for record in records if any(record["pinch_clamping_intervals"])
                    ]
            result[phase] = summary
        return result

    def lines(self):
        """The summary by phase as text lines, lengths in millimetres and forces in millinewtons."""

        def mm(value):
            return "     -" if value is None else f"{1e3 * value:6.2f}"

        lines = [
            "phase        status                  frames  cmd depth  jaw depth  wrist   shaft   dimple  force on tissue  "
            "force on tool  rejected  failed  step ms (median/max)  capsule inside: yielding / held / centre line [um]  "
            "actuator demand / limit  drawn jaws (CAD) inside [um]",
        ]
        for phase, item in self.phases().items():
            if "jaw_depth_max_m" not in item:
                lines.append(f"{item['number']} {phase:10s} {item['status']:23s} {item['frames']:6d}")
                continue
            lines.append(
                f"{item['number']} {phase:10s} {item['status']:23s} {item['frames']:6d}  "
                f"{mm(item['commanded_depth_max_m'])} mm  {mm(item['jaw_depth_max_m'])} mm  "
                f"{mm(item['wrist_depth_max_m'])}  {mm(item['shaft_depth_max_m'])}  {mm(item['dimple_depth_max_m'])}  "
                f"{1e3 * item['tissue_force_max_N']:12.2f} mN  {1e3 * item['tool_force_max_N']:10.2f} mN  "
                f"{item['rejected_intervals']:8d}  {item['failed_intervals']:6d}  "
                f"{item['step_ms_median']:7.2f} / {item['step_ms_max']:.2f}  "
                f"{1e6 * item['capsule_yielding_depth_max_m']:8.1f} / {1e6 * item['capsule_held_depth_max_m']:.1f} / "
                f"{1e6 * item['capsule_centre_depth_max_m']:.1f}  {item['actuator_demand_over_limit_max']:.2f}  "
                f"{1e6 * item['cad_jaw_depth_max_m']:.1f}"
            )
            for hold in item["holds"]:
                lines.append(
                    f"    end of {hold['segment']:5s} (frame {hold['frame']}): commanded {mm(hold['commanded_depth_m'])} mm, "
                    f"jaws {mm(hold['jaw_depth_m'])} mm inside the deformed surface (capsules "
                    f"{1e6 * hold['jaw_capsule_depth_m']:.1f} um, drawn jaws {1e6 * hold['cad_jaw_depth_m']:.1f} um), "
                    f"dimple {mm(hold['dimple_depth_m'])} mm, "
                    f"force {1e3 * hold['tissue_force_N']:.2f} mN, palm {1e3 * hold['palm_position_error_m']:.4f} mm from its target"
                    + (
                        f", jaws open {hold['jaw_half_opening_rad'][0]:.4f} / {hold['jaw_half_opening_rad'][1]:.4f} rad "
                        f"of {hold['commanded_half_opening_rad']:.4f} commanded, {hold['held_nodes']} nodes held"
                        if phase == PINCH
                        else ""
                    )
                    + (
                        f"; pinch {hold['pinch_state']}: carried {hold['pinch_carried_nodes'][0]} + "
                        f"{hold['pinch_carried_nodes'][1]}, clamped {hold['pinch_clamped_nodes'][0]} + "
                        f"{hold['pinch_clamped_nodes'][1]}, squeeze {1e3 * hold['pinch_squeeze_N']:.1f} mN, pull "
                        f"{1e3 * hold['pinch_pull_N']:.1f} mN, limit {1e3 * hold['pinch_limit_N']:.1f} mN"
                        if phase == PINCH and hold.get("pinch_state") is not None
                        else ""
                    )
                )
            if "pinch_clamping_intervals" in item:
                frames = item["pinch_clamping_frames"]
                lines.append(
                    f"    pinch, intervals per tool in which its jaws clamped tissue: {item['pinch_clamping_intervals']} "
                    f"(sticking {item['pinch_sticking_intervals']}, slipping {item['pinch_slipping_intervals']})"
                    + (f", frames {frames[0]} to {frames[-1]}" if frames else "")
                )
            for failure in item["failures"]:
                lines.append(
                    f"    FAILED in frame {failure['frame']} ({failure['segment']}), commanded {mm(failure['commanded_depth_m'])} mm, "
                    f"jaws last {mm(failure['last_jaw_depth_m'])} mm inside: {failure['reason']}"
                )
        return lines

    def report(self, **extra):
        """Everything measured, as a dictionary that ``json.dumps`` accepts."""
        return _clean(
            {
                "scene": self.scene.describe(),
                "path": {
                    **self.path.derivation,
                    "segments": [
                        {
                            "phase": segment.phase,
                            "segment": segment.label,
                            "first_frame": segment.first_frame,
                            "frames": segment.frames,
                            "commanded_depth_m": segment.depth,
                            "speed_m_s": segment.speed,
                            "hold": segment.hold,
                        }
                        for segment in self.path.segments
                    ],
                },
                "phases": self.phases(),
                "failures": self.failures,
                **extra,
                "frames": self.records,
                "intervals": self.intervals,
            }
        )

    def write(self, path, **extra):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.report(**extra), file, indent=1)
            file.write("\n")
