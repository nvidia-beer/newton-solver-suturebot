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


@dataclass(frozen=True)
class GrasperAssembly:
    """Builder identities and exact local collision capsules (SI units).

    Capsule rows zero/one are the palm shaft/wrist; rows two/three are jaws.
    Every capsule endpoint is expressed in its corresponding rigid body frame.
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
        """
        pose = _pose(palm_pose)
        aperture = self.config.initial_opening_rad if opening_rad is None else float(opening_rad)
        if not np.isfinite(aperture) or not 0 <= aperture <= self.config.maximum_opening_rad:
            raise ValueError("Reset jaw aperture is outside its physical range")
        coordinates = state.joint_q.numpy().copy()
        velocities = state.joint_qd.numpy().copy()
        q_starts = model.joint_q_start.numpy()
        v_starts = model.joint_qd_start.numpy()
        start = int(q_starts[self.palm_joint])
        coordinates[start : start + 7] = pose
        start = int(v_starts[self.palm_joint])
        velocities[start : start + 6] = 0
        coordinates[list(self.jaw_q_indices)] = (aperture / 2, -aperture / 2)
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


def _pose(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:]) - 1) > 1e-5:
        raise ValueError("A finite seven-component pose with unit quaternion is required")
    return value


def needle_to_gripper_pose(geometry, *, arc_fraction=0.8, config=None, tangent_roll_rad=0.0, bite_angle_rad=0.0):
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
    rotation = np.column_stack((x, y, z))
    quaternion = np.asarray(wp.quat_from_matrix(wp.mat33(rotation)))
    return np.r_[point - 0.006 * y, quaternion]


def add_grasper(builder, *, config=None, label="grasper"):
    """Append the authored PSM jaw capsule geometry with independent hinges.

    The palm aggregates the roll/pitch/yaw at their zero relative angles. Its
    free-body demonstration inertia is positive and explicit; it is not a
    dynamically calibrated reduction of the original seven-axis PSM chain.
    Jaw masses/inertias and wrist/jaw capsule dimensions match
    suturebot_rig.usda. The shaft is shortened from its proximal end to the
    configured centreline length; its radius and distal endpoint stay fixed.
    Palm mass/inertia remain the explicit uncalibrated demonstration values.
    """
    config = config or GrasperConfig()
    if "mujoco:condim" not in builder.custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=0,
        mu=config.friction,
        mu_torsional=config.torsional_friction_m,
        mu_rolling=config.rolling_friction_m,
        margin=0,
        gap=0,
        has_particle_collision=False,
    )
    palm = builder.add_link(
        mass=0.112,
        inertia=wp.mat33(np.diag([0.002504, 6e-6, 0.002504])),
        lock_inertia=True,
        label=f"{label}/palm",
    )
    palm_joint = builder.add_joint_free(child=palm, label=f"{label}/free")
    endpoints = np.asarray(
        [
            [[0, -0.0121 - config.shaft_length_m, 0], [0, -0.0121, 0]],
            [[0, -0.0091, 0], [0, -0.0011, 0]],
            [[-0.00075, 0.0015, 0], [-0.00075, 0.009, 0]],
            [[0.00075, 0.0015, 0], [0.00075, 0.009, 0]],
        ]
    )
    radii = np.asarray([0.0043, 0.0022, 0.00065, 0.00065])

    def capsule(body, row, suffix):
        start, end = endpoints[row]
        return builder.add_shape_capsule(
            body,
            xform=wp.transform((start + end) / 2, wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2)),
            radius=float(radii[row]),
            half_height=float(np.linalg.norm(end - start) / 2),
            cfg=shape_cfg,
            label=f"{label}/{suffix}",
            custom_attributes={
                "mujoco:condim": 6,
                "mujoco:geom_priority": 1,
                "mujoco:solref_mode": 1,
                "mujoco:solref": wp.vec2(config.contact_time_constant_s, config.contact_damping_ratio),
                "mujoco:geom_solimp": [config.contact_impedance, config.contact_impedance, 0.001, 0.5, 2.0],
            },
        )

    shapes = [capsule(palm, 0, "shaft"), capsule(palm, 1, "wrist")]
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
        shapes.append(capsule(body, side + 2, f"jaw_capsule_{side}"))
        bodies.append(body)
        joints.append(joint)
    builder.add_articulation([palm_joint, *joints], label=label)
    return GrasperAssembly(
        config,
        palm,
        tuple(bodies),
        palm_joint,
        tuple(joints),
        tuple(q_indices),
        tuple(dof_indices),
        (palm, palm, *bodies),
        tuple(shapes),
        endpoints,
        radii,
    )
