# SPDX-License-Identifier: Apache-2.0
"""A free palm and two force-limited hinged jaws for native MuJoCo grasping.

The needle remains a separate free body. Only frictional collision contacts
connect it to this tool; this module creates no weld, equality, or needle drive.
Dimensions, inertial values and gains are demonstration parameters, not a
calibrated surgical instrument model.
"""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo


@dataclass(frozen=True)
class GrasperConfig:
    # Centreline length; rounded capsule caps add one radius at each end.
    shaft_length_m: float = 0.12
    maximum_opening_rad: float = 1.3
    initial_opening_rad: float = 0.06
    jaw_stiffness_nm_rad: float = 0.08
    jaw_damping_nms_rad: float = 0.004
    jaw_torque_limit_nm: float = 0.02
    friction: float = 1.0
    torsional_friction_m: float = 2.8e-4
    rolling_friction_m: float = 2.0e-5
    contact_time_constant_s: float = 0.008
    contact_damping_ratio: float = 1.0
    contact_impedance: float = 0.9999
    # Friction coefficient between the jaws and tissue that they clamp (the pinch of the tool/tissue
    # exchange). Source: Heijnsdijk, de Visser, Dankelman, Gouma, "Slip and damage properties of jaws of
    # laparoscopic graspers", Surg Endosc 18(6):974-979, 2004: bowel pulled with 5 N slips out of a grasper
    # between a pinch of 3 N and one of 22 N, by the jaw profile. With two faces, pull = 2 mu x pinch:
    # 5 / (2 x 22) = 0.11 to 5 / (2 x 3) = 0.83. The upper end, a profiled jaw, is taken: an effective
    # number (a profile holds by interlock).
    tissue_friction: float = 0.83

    def __post_init__(self):
        values = np.asarray(list(vars(self).values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Grasper parameters must be finite and nonnegative")
        if not 0 < self.shaft_length_m <= 0.547:
            raise ValueError("Shaft centreline length must be positive and at most the source CAD length of 0.547 m")
        if min(self.maximum_opening_rad, self.jaw_stiffness_nm_rad, self.jaw_torque_limit_nm) <= 0:
            raise ValueError("Opening, actuator stiffness and torque limit must be positive")
        if self.initial_opening_rad > self.maximum_opening_rad:
            raise ValueError("Initial grasper opening exceeds its maximum")
        if min(self.contact_time_constant_s, self.contact_damping_ratio) <= 0 or not 0 < self.contact_impedance < 1:
            raise ValueError("Contact time/damping must be positive and impedance strictly between zero and one")


# Name of the tool part that each drawn part of the CAD stands for in the capsule set.
CAPSULE_PART_NAMES = {"roll": "shaft", "pitch": "wrist", "jaw1": "jaw_1", "jaw2": "jaw_2"}


@dataclass(frozen=True)
class GrasperAssembly:
    """Builder identities and exact local collision capsules (SI units).

    The capsules are the one capsule set of the tool, fitted to the CAD that is drawn on the
    same bodies (``_dvrk_capsules.dvrk_tool_capsules``): the rigid collision shapes of the
    MuJoCo model, the tool/tissue contact, the thread's contact with the tools, the
    measurements and the collision shapes of the viewer all read these rows. ``capsule_parts``
    names the part of each row: "shaft", "wrist", "jaw_1" or "jaw_2"; a part has one or more
    rows. Every capsule endpoint is expressed in its corresponding rigid body frame.
    ``capsule_bounds`` is, for each row, how far a point of the drawn part can be outside it [m].
    Jaw coordinates are signed hinge angles in radians, independently solved.
    """

    config: GrasperConfig
    palm_body: int
    jaw_bodies: tuple[int, int]
    palm_joint: int
    jaw_joints: tuple[int, int]
    jaw_q_indices: tuple[int, int]
    jaw_dof_indices: tuple[int, int]
    capsule_bodies: tuple[int, ...]
    capsule_shapes: tuple[int, ...]
    capsule_endpoints: np.ndarray
    capsule_radii: np.ndarray
    capsule_parts: tuple[str, ...]
    capsule_bounds: np.ndarray

    def set_aperture(self, control, normalized_aperture):
        """Set finite native actuator targets; never assign a physical state."""
        value = float(normalized_aperture)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Normalized jaw aperture must be in [0, 1]")
        target = 0.5 * self.config.maximum_opening_rad * value
        # Newton position targets use q-coordinate indices (FREE has 7),
        # whereas velocity targets and efforts use DOF indices (FREE has 6).
        for index, sign in zip(self.jaw_q_indices, (1, -1), strict=True):
            control.joint_target_q[index : index + 1].fill_(sign * target)

    def initialize(self, model, state, palm_pose, *, opening_rad=None):
        """Reset-only placement through joint coordinates and forward kinematics.

        Other articulations, including the independent needle, retain their
        coordinates. The caller owns reset lifecycle/native solver reset.
        ``opening_rad`` is the total opening of the jaws, which then open
        symmetrically, or one opening angle per jaw (``jaw_opening_on_needle``);
        by default the configured initial opening.
        """
        pose = _pose(palm_pose)
        jaw_angles = _jaw_angles(self.config, opening_rad)
        coordinates = state.joint_q.numpy().copy()
        velocities = state.joint_qd.numpy().copy()
        q_starts = model.joint_q_start.numpy()
        v_starts = model.joint_qd_start.numpy()
        start = int(q_starts[self.palm_joint])
        coordinates[start : start + 7] = pose
        start = int(v_starts[self.palm_joint])
        velocities[start : start + 6] = 0
        coordinates[list(self.jaw_q_indices)] = jaw_angles
        velocities[list(self.jaw_dof_indices)] = 0
        state.joint_q.assign(coordinates)
        state.joint_qd.assign(velocities)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def observe(self, model, state, solver, needle_body, *, host_state=None):
        """Actual diagnostics, optionally sharing the caller's accepted-state readback."""
        import mujoco

        if host_state is None:
            host_state = {name: getattr(state, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
        poses = host_state["body_q"].astype(float)
        velocities = host_state["body_qd"].astype(float)
        normal = np.zeros(2)
        counts = np.zeros(2, dtype=int)
        # Native body IDs are resolved by labels, rather than assumed offsets.
        needle_native = mujoco.mj_name2id(
            solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, model.body_label[needle_body].replace("/", "_")
        )
        native_jaws = [
            mujoco.mj_name2id(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, model.body_label[body].replace("/", "_"))
            for body in self.jaw_bodies
        ]
        if needle_native < 0 or min(native_jaws) < 0:
            raise ValueError("Native MuJoCo rigid body labels do not match the grasp assembly")
        if getattr(solver, "accepted_grasp_metrics", None) is not None:
            metrics = solver.accepted_grasp_metrics.numpy()[0]
            normal = metrics[:2].astype(float)
            counts = metrics[2:].astype(int)
        else:
            wrench = np.empty(6)
            for index in range(solver.mj_data.ncon):
                contact = solver.mj_data.contact[index]
                bodies = solver.mj_model.geom_bodyid[[contact.geom1, contact.geom2]]
                if needle_native not in bodies:
                    continue
                for jaw, body in enumerate(native_jaws):
                    if body in bodies:
                        mujoco.mj_contactForce(solver.mj_model, solver.mj_data, index, wrench)
                        normal[jaw] += max(0.0, wrench[0])
                        counts[jaw] += int(contact.efc_address >= 0 and wrench[0] > 0)
        relative = wp.transform_multiply(
            wp.transform_inverse(wp.transform(*poses[self.palm_body])), wp.transform(*poses[needle_body])
        )
        return {
            "gripper_pose": poses[self.palm_body].copy(),
            "gripper_velocity": velocities[self.palm_body].copy(),
            "jaw_position_rad": host_state["joint_q"][list(self.jaw_q_indices)].astype(float),
            "jaw_velocity_rad_s": host_state["joint_qd"][list(self.jaw_dof_indices)].astype(float),
            "jaw_body_poses": poses[list(self.jaw_bodies)].copy(),
            "jaw_body_velocities": velocities[list(self.jaw_bodies)].copy(),
            "jaw_contact_normal_force_n": normal,
            "jaw_needle_contact_count": counts,
            "needle_relative_pose": np.asarray(relative, dtype=float),
        }


def _segment_nearest(first, second):
    """Distance between two straight segments [m] (each two end points, shape (2, 3)) and where on the first it is taken.

    Returns:
        The distance and the fraction of the first segment, from its first end point, of its nearest point.
    """
    (p, a), (q, b) = (np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    u, v, w = a - p, b - q, p - q
    uu, uv, vv, uw, vw = u @ u, u @ v, v @ v, u @ w, v @ w
    best = (np.inf, 0.0)
    # The nearest points lie inside both segments, or one of them is an end point.
    denominator = uu * vv - uv * uv
    if denominator > 1e-18 * max(uu * vv, 1e-300):
        s, t = (uv * vw - vv * uw) / denominator, (uu * vw - uv * uw) / denominator
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            best = (float(np.linalg.norm(w + s * u - t * v)), float(s))
    for fraction_first, point in ((0.0, p), (1.0, a)):
        fraction = 0.0 if vv <= 0.0 else float(np.clip((point - q) @ v / vv, 0.0, 1.0))
        best = min(best, (float(np.linalg.norm(point - q - fraction * v)), fraction_first))
    for point in (q, b):
        fraction = 0.0 if uu <= 0.0 else float(np.clip((point - p) @ u / uu, 0.0, 1.0))
        best = min(best, (float(np.linalg.norm(point - p - fraction * u)), fraction))
    return best


def _segment_distance(first, second):
    """Distance between two straight segments [m], each given by its two end points, shape (2, 3)."""
    return _segment_nearest(first, second)[0]


def jaw_opening_on_needle(assembly, centerline, needle_radius, *, tolerance_rad=1.0e-7):
    """Hinge angles at which the two jaws are seated on a needle that lies between them [rad].

    A reset places the needle in the bite and the jaws about it. Where they are seated on it
    follows from the capsule set of the tool. For each capsule of a jaw, the hinge angle at
    which the gap between it and the needle's capsules is zero is found by bisection between
    the closed jaw and the largest opening (the closed end of the last interval: an overlap of
    less than ``tolerance_rad`` times the jaw's length). The capsule with the largest such
    angle is the one that meets the needle first as the jaw closes. The jaw is seated when
    every capsule that stands beside that one where it meets the needle (its axis passes that
    place along the jaw) touches the needle too: at the smallest of their angles. The capsules
    that met the wire before the last lie inside it by the difference, a few micrometres,
    which the contact resolves in its first steps; every contact of the bite exists from the
    first step on.

    Args:
        assembly: The tool (``GrasperAssembly``); its jaws turn about the z axis of its palm.
        centerline: Points of the needle's centreline in the frame of the palm [m], shape (count, 3).
        needle_radius: Radius of the needle wire [m].
        tolerance_rad: Width of the last bisection interval [rad].

    Returns:
        The opening angle of each jaw from its closed position [rad], both nonnegative, as
        ``initialize`` takes them; zero for a jaw whose capsules do not reach the needle when
        it is closed.
    """
    centerline = np.asarray(centerline, dtype=float).reshape(-1, 3)
    segments = np.stack((centerline[:-1], centerline[1:]), axis=1)
    limit = 0.5 * assembly.config.maximum_opening_rad
    angles = []
    for body, sign in zip(assembly.jaw_bodies, (1.0, -1.0), strict=True):
        rows = [row for row, capsule_body in enumerate(assembly.capsule_bodies) if capsule_body == body]

        def turned(row, angle, sign=sign):
            cosine, sine = np.cos(sign * angle), np.sin(sign * angle)
            turn = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
            return assembly.capsule_endpoints[row] @ turn.T

        def gap(row, angle):
            ends = turned(row, angle)
            return (
                min(_segment_distance(ends, segment) for segment in segments)
                - assembly.capsule_radii[row]
                - needle_radius
            )

        touching = {}
        for row in rows:
            if gap(row, 0.0) >= 0.0:
                continue
            if gap(row, limit) < 0.0:
                raise ValueError("The needle does not fit between the fully opened jaws")
            low, high = 0.0, limit
            while high - low > tolerance_rad:
                middle = 0.5 * (low + high)
                low, high = (middle, high) if gap(row, middle) < 0.0 else (low, middle)
            touching[row] = low
        if not touching:
            angles.append(0.0)
            continue
        first = max(touching, key=touching.get)
        # Where along the jaw the first capsule meets the needle, in the frame of the jaw.
        ends = turned(first, touching[first])
        fraction = min(_segment_nearest(ends, segment) for segment in segments)[1]
        start, end = assembly.capsule_endpoints[first]
        along = (end - start) / np.linalg.norm(end - start)
        place = start + fraction * (end - start)
        beside = [
            row
            for row in touching
            if ((assembly.capsule_endpoints[row, 0] - place) @ along)
            * ((assembly.capsule_endpoints[row, 1] - place) @ along)
            <= 0.0
        ]
        angles.append(min(touching[row] for row in beside))
    return tuple(angles)


def _jaw_angles(config, opening_rad):
    """Hinge coordinates of the two jaws for a reset: from a total opening, or from one opening angle per jaw [rad]."""
    if opening_rad is None:
        opening_rad = config.initial_opening_rad
    angles = np.asarray(opening_rad, dtype=float)
    angles = np.full(2, 0.5 * float(angles)) if angles.ndim == 0 else angles
    if angles.shape != (2,) or not np.isfinite(angles).all() or np.any(angles < 0):
        raise ValueError("Reset jaw aperture is outside its physical range")
    if np.any(angles > 0.5 * config.maximum_opening_rad):
        raise ValueError("Reset jaw aperture is outside its physical range")
    return angles[0], -angles[1]


def _pose(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:]) - 1) > 1e-5:
        raise ValueError("A finite seven-component pose with unit quaternion is required")
    return value


def needle_to_gripper_pose(
    geometry, *, arc_fraction=0.8, config=None, tangent_roll_rad=0.0, bite_angle_rad=0.0, jaws_exchanged=False
):
    """Nominal tool frame relative to the needle, for reset/desired commands.

    This transform is never enforced as a constraint. By default the jaws
    close normal to the needle plane, with the needle tangent along local Z.
    ``tangent_roll_rad`` twists the tool's X/Y axes about that tangent while
    preserving the bite point at local (0, 0.006, 0). This lets the articulated
    wrist approach the same grasp from a different orientation.
    ``bite_angle_rad`` then rotates about the resulting jaw-closing X axis,
    keeping the same bite point and closing normal while tilting the tangent
    within the jaws' Y/Z plane. Its magnitude must be less than pi/2 so the
    needle still crosses the jaws rather than lying along their length.
    ``jaws_exchanged`` then turns the tool half a turn about its own jaw axis
    (local Y): the bite point and the jaw axis stay, the closing axis X and
    the axis Z are reversed, so each jaw closes from the side where the other
    one was. It is the same grasp of the needle; a fixed-trocar tool reaches
    it with its roll joint half a turn on and both wrist angles reversed.
    """
    config = config or GrasperConfig()
    if not np.isfinite(arc_fraction) or not 0 < arc_fraction < 1:
        raise ValueError("Grasp location must lie strictly inside the needle arc")
    if not np.isfinite(tangent_roll_rad):
        raise ValueError("The grasp twist about the needle tangent must be finite")
    if not np.isfinite(bite_angle_rad) or not -np.pi / 2 < bite_angle_rad < np.pi / 2:
        raise ValueError("The grasp bite angle must be finite and strictly between -pi/2 and pi/2")
    points = np.asarray(geometry.centerline, dtype=float)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    distance = arc_fraction * distances[-1]
    index = min(np.searchsorted(distances, distance, side="right") - 1, len(points) - 2)
    fraction = (distance - distances[index]) / (distances[index + 1] - distances[index])
    point = (1 - fraction) * points[index] + fraction * points[index + 1]
    z = points[index + 1] - points[index]
    z /= np.linalg.norm(z)
    x = np.asarray(geometry.arc_normal, dtype=float).copy()
    x -= z * (z @ x)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if tangent_roll_rad != 0:
        cosine, sine = np.cos(tangent_roll_rad), np.sin(tangent_roll_rad)
        x, y = cosine * x + sine * y, -sine * x + cosine * y
    if bite_angle_rad != 0:
        cosine, sine = np.cos(bite_angle_rad), np.sin(bite_angle_rad)
        y, z = cosine * y + sine * z, -sine * y + cosine * z
    if jaws_exchanged:
        x, z = -x, -z
    rotation = np.column_stack((x, y, z))
    quaternion = np.asarray(wp.quat_from_matrix(wp.mat33(rotation)))
    return np.r_[point - 0.006 * y, quaternion]


def tool_instrument(label, instrument=None):
    """The instrument whose CAD a tool model is fitted to: the one asked for, else the label if it names one, else PSM1."""
    if instrument is None:
        instrument = label if label in ("PSM1", "PSM2") else "PSM1"
    if instrument not in ("PSM1", "PSM2"):
        raise ValueError("The instrument of a dVRK tool is PSM1 or PSM2")
    return instrument


def add_tool_capsules(builder, capsules, part_bodies, config, label, *, part_frames=None):
    """Add the capsule set of a tool to its bodies as the collision shapes of the rigid model.

    The capsules of the wrist and of the two jaws stand for adjoining pieces of one mechanism
    and overlap where those pieces meet: the roots of the jaws lie in the fork of the wrist
    and side by side along their hinge axis, and the capsules of a blade reach across the
    plane in which the drawn jaws meet by the share of the edges of its gripping face. The
    hinge limits keep each jaw on its side of that plane. The pairs among wrist and jaws are
    therefore excluded from the collision; shaft/jaw, tool/tool and external contacts remain.

    Args:
        builder: Model builder that holds the bodies.
        capsules: The ``ToolCapsules`` of the instrument.
        part_bodies: Body of each drawn part: keys "roll", "pitch", "jaw1", "jaw2".
        config: The ``GrasperConfig`` of the tool.
        label: Label of the tool; a shape is called ``<label>/<part>_capsule_<row of the part>``.
        part_frames: For a part whose capsules are not given in the frame of its body, the pose
            of the part in that frame (4 x 4).

    Returns:
        Body, shape, end points in the body frame [m], radius [m], part name and bound [m] of
        every capsule: the six capsule fields of a ``GrasperAssembly``.
    """
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=0,
        mu=config.friction,
        mu_torsional=config.torsional_friction_m,
        mu_rolling=config.rolling_friction_m,
        margin=0,
        gap=0,
        has_particle_collision=False,
    )
    bodies, shapes, endpoints, parts, counts = [], [], [], [], {}
    for part, ends, radius in zip(capsules.parts, capsules.endpoints, capsules.radii, strict=True):
        frame = None if part_frames is None else part_frames.get(part)
        if frame is not None:
            ends = ends @ np.asarray(frame)[:3, :3].T + np.asarray(frame)[:3, 3]
        start, end = (np.asarray(point, dtype=float) for point in ends)
        name = CAPSULE_PART_NAMES[part]
        row = counts[name] = counts.get(name, -1) + 1
        shapes.append(
            builder.add_shape_capsule(
                part_bodies[part],
                xform=wp.transform(
                    wp.vec3(*((start + end) / 2)),
                    wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(end - start))),
                ),
                radius=float(radius),
                half_height=float(np.linalg.norm(end - start) / 2),
                cfg=shape_cfg,
                label=f"{label}/{name}_capsule_{row}",
                custom_attributes={
                    "mujoco:condim": 6,
                    "mujoco:geom_priority": 1,
                    "mujoco:solref_mode": 1,
                    "mujoco:solref": wp.vec2(config.contact_time_constant_s, config.contact_damping_ratio),
                    "mujoco:geom_solimp": [config.contact_impedance, config.contact_impedance, 0.001, 0.5, 2.0],
                },
            )
        )
        bodies.append(part_bodies[part])
        endpoints.append(np.stack((start, end)))
        parts.append(name)
    distal = [(shape, name) for shape, name in zip(shapes, parts, strict=True) if name != "shaft"]
    for index, (first, first_name) in enumerate(distal):
        for second, second_name in distal[index + 1 :]:
            if first_name != second_name:
                builder.add_shape_collision_filter_pair(first, second)
    return (
        tuple(bodies),
        tuple(shapes),
        np.asarray(endpoints),
        np.asarray(capsules.radii, dtype=float).copy(),
        tuple(parts),
        np.asarray(capsules.bounds, dtype=float).copy(),
    )


def add_grasper(builder, *, config=None, label="grasper", instrument=None):
    """Append the jaws of a PSM tool on a free palm, with independent hinges.

    The palm aggregates the roll/pitch/yaw at their zero relative angles. Its
    free-body demonstration inertia is positive and explicit; it is not a
    dynamically calibrated reduction of the original seven-axis PSM chain.
    Jaw masses/inertias match suturebot_rig.usda. The collision capsules are the
    capsule set of the instrument, fitted to its drawn CAD with the shaft of the
    configured centreline length (``dvrk_tool_capsules``); shaft and wrist lie on
    the palm. ``instrument`` is "PSM1" or "PSM2"; by default the label if it is
    one of them, else "PSM1".
    Palm mass/inertia remain the explicit uncalibrated demonstration values.
    """
    from ._dvrk_capsules import dvrk_tool_capsules

    config = config or GrasperConfig()
    if "mujoco:condim" not in builder.custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    palm = builder.add_link(
        mass=0.112,
        inertia=wp.mat33(np.diag([0.002504, 6e-6, 0.002504])),
        lock_inertia=True,
        label=f"{label}/palm",
    )
    palm_joint = builder.add_joint_free(child=palm, label=f"{label}/free")
    bodies, joints, q_indices, dof_indices = [], [], [], []
    for side, sign in enumerate((1, -1)):
        body = builder.add_link(
            mass=0.006,
            inertia=wp.mat33(np.eye(3) * 2e-6),
            lock_inertia=True,
            label=f"{label}/jaw_{side}",
        )
        joint = builder.add_joint_revolute(
            palm,
            body,
            axis=(0, 0, 1),
            target_q=sign * config.initial_opening_rad / 2,
            target_ke=config.jaw_stiffness_nm_rad,
            target_kd=config.jaw_damping_nms_rad,
            limit_lower=0 if sign > 0 else -config.maximum_opening_rad / 2,
            limit_upper=config.maximum_opening_rad / 2 if sign > 0 else 0,
            effort_limit=config.jaw_torque_limit_nm,
            actuator_mode=newton.JointTargetMode.POSITION,
            label=f"{label}/hinge_{side}",
        )
        q_indices.append(builder.joint_q_start[joint])
        dof_indices.append(builder.joint_qd_start[joint])
        builder.joint_q[q_indices[-1]] = sign * config.initial_opening_rad / 2
        bodies.append(body)
        joints.append(joint)
    builder.add_articulation([palm_joint, *joints], label=label)
    capsules = dvrk_tool_capsules(tool_instrument(label, instrument), float(config.shaft_length_m))
    return GrasperAssembly(
        config,
        palm,
        tuple(bodies),
        palm_joint,
        tuple(joints),
        tuple(q_indices),
        tuple(dof_indices),
        *add_tool_capsules(
            builder,
            capsules,
            {"roll": palm, "pitch": palm, "jaw1": bodies[0], "jaw2": bodies[1]},
            config,
            label,
            # The palm is the wrist-yaw frame: shaft and wrist are placed in it as the CAD assembles them.
            part_frames={name: capsules.frames[name] for name in ("roll", "pitch")},
        ),
    )
