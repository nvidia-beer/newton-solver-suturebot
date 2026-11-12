# SPDX-License-Identifier: Apache-2.0
"""Fixed-trocar PSM surrogate with native bounded joint actuators.

Joint frames follow the bundled dVRK rig. The short shaft, fixture placement,
inertias and actuator gains are illustrative rather than calibrated hardware.
Only the separate needle has a free joint; no weld or equality holds it.
"""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from .grasp import GrasperAssembly, GrasperConfig, _pose

ARM_JOINT_NAMES = ("rcm_yaw", "rcm_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
NOMINAL_INSERTION_M = 0.08
WRIST_LENGTH_M = 0.0091
_WRIST_ROTATION = (-0.5, -0.5, -0.5, 0.5)  # xyzw, source USD (w,x,y,z)=(.5,-.5,-.5,-.5).
_NOMINAL_ARM_POSITIONS = (0.0, 0.0, 0.0, 0.0, -0.7, 0.0)


def _arm_pose(positions, nominal_insertion_m=NOMINAL_INSERTION_M):
    """Wrist-yaw pose relative to the fixed RCM frame, from exact joint frames."""
    yaw, pitch, insertion, roll, wrist_pitch, wrist_yaw = positions

    def rotate(axis, angle):
        return wp.transform((0, 0, 0), wp.quat_from_axis_angle(wp.vec3(axis), float(angle)))

    transforms = (
        rotate((0, 1, 0), yaw),
        rotate((1, 0, 0), pitch),
        wp.transform((0, 0, nominal_insertion_m + insertion), wp.quat_identity()),
        rotate((0, 0, 1), roll),
        wp.transform((0, 0, 0), wp.quat(*_WRIST_ROTATION)),
        rotate((0, 0, 1), wrist_pitch),
        wp.transform((WRIST_LENGTH_M, 0, 0), wp.quat(*_WRIST_ROTATION)),
        rotate((0, 0, 1), wrist_yaw),
    )
    result = wp.transform_identity()
    for transform in transforms:
        result = wp.transform_multiply(result, transform)
    return result


@dataclass(frozen=True)
class RCMGrasperAssembly(GrasperAssembly):
    """Actual articulated identities; the palm is the wrist-yaw body.

    ``set_anchor`` and ``initialize`` are reset-only setup operations. Construct
    the native rigid solver after setting the anchor; its compiled fixed-base
    transforms must never be changed to satisfy a runtime target.
    """

    arm_joints: tuple[int, ...]
    arm_q_indices: tuple[int, ...]
    arm_dof_indices: tuple[int, ...]
    arm_lower: np.ndarray
    arm_upper: np.ndarray
    nominal_arm_positions: np.ndarray
    nominal_insertion_m: float
    insertion_limits_m: tuple[float, float]
    part_body_indices: dict[str, int]
    body_indices: tuple[int, ...]
    trocar_pose: np.ndarray
    _anchor_initialized: bool = False

    def set_anchor(self, model, palm_pose, *, reset=False):
        """Place the fixed trocar from the nominal articulated wrist pose.

        A caller that has verified physical time is zero may explicitly pass
        ``reset=True`` to replace an earlier preview placement. It must rebuild
        the native solver afterwards. Runtime IK changes joint targets only.
        """
        pose = _pose(palm_pose)
        anchor = np.asarray(
            wp.transform_multiply(
                wp.transform(*pose),
                wp.transform_inverse(_arm_pose(self.nominal_arm_positions, self.nominal_insertion_m)),
            ),
            dtype=float,
        )
        self.set_trocar_pose(model, anchor, reset=reset)

    def set_trocar_pose(self, model, anchor_pose, *, reset=False):
        """Author an explicit fixed world trocar pose before solver creation.

        Replacing a preview requires ``reset=True`` after the caller verifies
        physical time is zero, followed by rebuilding the native rigid solver.
        Runtime IK must leave this transform unchanged.
        """
        anchor = _pose(anchor_pose).copy()
        if self._anchor_initialized and not reset:
            raise RuntimeError("The fixed trocar is already placed; rebuild the assembly to change it")
        anchor[3:] /= np.linalg.norm(anchor[3:])
        transforms = model.joint_X_p.numpy().copy()
        transforms[self.arm_joints[0]] = anchor
        model.joint_X_p.assign(transforms)
        self.trocar_pose[:] = anchor
        object.__setattr__(self, "_anchor_initialized", True)

    def initialize(self, model, state, palm_pose=None, *, opening_rad=None, arm_positions=None):
        """Reset scalar arm/jaw coordinates; preserve the independent needle.

        A reset-time IK solution may supply ``arm_positions`` for an explicitly
        placed trocar. Runtime commands use ``set_arm_targets`` instead.
        """
        pose = None if palm_pose is None else _pose(palm_pose)
        if not self._anchor_initialized:
            raise RuntimeError("Place the fixed trocar before initializing its joint coordinates")
        opening = self.config.initial_opening_rad if opening_rad is None else float(opening_rad)
        if not np.isfinite(opening) or not 0 <= opening <= self.config.maximum_opening_rad:
            raise ValueError("Reset jaw aperture is outside its physical range")
        positions = self.nominal_arm_positions if arm_positions is None else np.asarray(arm_positions, dtype=float)
        self._validate_arm_positions(positions)
        expected = np.asarray(
            wp.transform_multiply(wp.transform(*self.trocar_pose), _arm_pose(positions, self.nominal_insertion_m)),
            dtype=float,
        )
        if pose is not None and (
            np.linalg.norm(expected[:3] - pose[:3]) > 1e-6 or abs(np.dot(expected[3:], pose[3:])) < 1 - 1e-6
        ):
            raise ValueError("Reset pose does not match the fixed fixture and arm coordinates; solve IK first")
        coordinates, velocities = (state.joint_q.numpy().copy(), state.joint_qd.numpy().copy())
        coordinates[list(self.arm_q_indices)] = positions
        coordinates[list(self.jaw_q_indices)] = (opening / 2, -opening / 2)
        velocities[list((*self.arm_dof_indices, *self.jaw_dof_indices))] = 0
        state.joint_q.assign(coordinates)
        state.joint_qd.assign(velocities)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def set_arm_targets(self, control, positions):
        """Command finite joint-position actuators; reject unreachable limits."""
        values = np.asarray(positions, dtype=float)
        self._validate_arm_positions(values)
        for index, value in zip(self.arm_q_indices, values, strict=True):
            control.joint_target_q[index : index + 1].fill_(float(value))

    def _validate_arm_positions(self, values):
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("RCM arm targets must contain six finite joint coordinates")
        if np.any(values < self.arm_lower) or np.any(values > self.arm_upper):
            raise ValueError("RCM arm targets exceed joint limits")


def add_rcm_grasper(
    builder, *, config=None, label="grasper", nominal_insertion_m=NOMINAL_INSERTION_M, insertion_limits_m=(-0.03, 0.03)
):
    """Add two RCM hinges, axial slide, roll, two wrists and two jaws.

    The first hinge is attached to world, with no free palm or Cartesian pose
    constraint. The 80 mm nominal slide distance and +/-30 mm travel keep the
    fixed pivot inside the default 120 mm shaft. Custom insertion geometry
    must keep the pivot on the physical shaft throughout its travel. Dynamics
    and collision geometry share actual link frames.
    A -0.7 rad nominal wrist pitch leaves room for the insertion arc while
    preserving the original wrist limits; the fixed base follows exact FK.
    """
    config = config or GrasperConfig()
    nominal_insertion_m = float(nominal_insertion_m)
    insertion_limits = np.asarray(insertion_limits_m, dtype=float)
    if not np.isfinite(nominal_insertion_m) or nominal_insertion_m <= 0:
        raise ValueError("The nominal insertion distance must be finite and positive")
    if (
        insertion_limits.shape != (2,)
        or not np.isfinite(insertion_limits).all()
        or not insertion_limits[0] <= 0 <= insertion_limits[1]
        or insertion_limits[0] >= insertion_limits[1]
    ):
        raise ValueError("Insertion limits must be a finite increasing pair containing the nominal zero position")
    if (
        nominal_insertion_m + insertion_limits[0] <= 0.003
        or nominal_insertion_m + insertion_limits[1] > config.shaft_length_m + 0.003
    ):
        raise ValueError("The fixed trocar must remain on the shaft throughout the insertion range")
    if "mujoco:condim" not in builder.custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    names = ("rcm_yaw", "rcm_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw", "jaw_0", "jaw_1")
    bodies = []
    for index, name in enumerate(names):
        mass = 0.08 if index < 3 else (0.1 if index == 3 else 0.006)
        inertia = np.diag([0.0025, 0.0025, 2e-6]) if index == 3 else np.eye(3) * 2e-6
        bodies.append(
            builder.add_link(mass=mass, inertia=wp.mat33(inertia), lock_inertia=True, label=f"{label}/{name}")
        )
    lower = np.asarray([-0.6, -0.6, insertion_limits[0], -np.pi, -1.1, -1.1])
    upper = -lower
    upper[2] = insertion_limits[1]
    joints, q_indices, dof_indices = [], [], []
    axes = ((0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1))
    frames = [wp.transform_identity() for _ in range(6)]
    frames[2] = wp.transform((0, 0, nominal_insertion_m), wp.quat_identity())
    frames[4] = wp.transform((0, 0, 0), wp.quat(*_WRIST_ROTATION))
    frames[5] = wp.transform((WRIST_LENGTH_M, 0, 0), wp.quat(*_WRIST_ROTATION))
    for index, (name, axis, frame, kp, kd, effort) in enumerate(
        zip(
            ARM_JOINT_NAMES,
            axes,
            frames,
            (12, 12, 800, 2, 0.4, 0.4),
            (1, 1, 20, 0.1, 0.02, 0.02),
            (2, 2, 30, 2, 2, 2),
            strict=True,
        )
    ):
        add_joint = builder.add_joint_prismatic if name == "insertion" else builder.add_joint_revolute
        joint = add_joint(
            -1 if index == 0 else bodies[index - 1],
            bodies[index],
            parent_xform=frame,
            axis=axis,
            target_q=_NOMINAL_ARM_POSITIONS[index],
            target_ke=kp,
            target_kd=kd,
            limit_lower=float(lower[index]),
            limit_upper=float(upper[index]),
            effort_limit=effort,
            actuator_mode=newton.JointTargetMode.POSITION,
            label=f"{label}/{name}_joint",
        )
        joints.append(joint)
        q_indices.append(builder.joint_q_start[joint])
        dof_indices.append(builder.joint_qd_start[joint])
        builder.joint_q[q_indices[-1]] = _NOMINAL_ARM_POSITIONS[index]
    jaw_joints, jaw_q, jaw_dof = [], [], []
    for side, sign in enumerate((1, -1)):
        joint = builder.add_joint_revolute(
            bodies[5],
            bodies[6 + side],
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
        jaw_joints.append(joint)
        jaw_q.append(builder.joint_q_start[joint])
        jaw_dof.append(builder.joint_qd_start[joint])
        builder.joint_q[jaw_q[-1]] = sign * config.initial_opening_rad / 2
    builder.add_articulation([*joints, *jaw_joints], label=label)
    endpoints = np.asarray(
        [
            [[0, 0, -0.003 - config.shaft_length_m], [0, 0, -0.003]],
            [[0, 0, 0], [0.008, 0, 0]],
            [[-0.00075, 0.0015, 0], [-0.00075, 0.009, 0]],
            [[0.00075, 0.0015, 0], [0.00075, 0.009, 0]],
        ]
    )
    capsule_bodies = (bodies[3], bodies[4], bodies[6], bodies[7])
    radii = np.asarray([0.0043, 0.0022, 0.00065, 0.00065])
    shape_config = newton.ModelBuilder.ShapeConfig(
        density=0,
        mu=config.friction,
        mu_torsional=config.torsional_friction_m,
        mu_rolling=config.rolling_friction_m,
        margin=0,
        gap=0,
        has_particle_collision=False,
    )
    rotations = (
        wp.quat_identity(),
        wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2),
        wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2),
        wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2),
    )
    shapes = []
    for row, (body, rotation, name) in enumerate(
        zip(capsule_bodies, rotations, ("shaft", "wrist", "jaw_capsule_0", "jaw_capsule_1"), strict=True)
    ):
        start, end = endpoints[row]
        shapes.append(
            builder.add_shape_capsule(
                body,
                xform=wp.transform((start + end) / 2, rotation),
                radius=float(radii[row]),
                half_height=float(np.linalg.norm(end - start) / 2),
                cfg=shape_config,
                label=f"{label}/{name}",
                custom_attributes={
                    "mujoco:condim": 6,
                    "mujoco:geom_priority": 1,
                    "mujoco:solref_mode": 1,
                    "mujoco:solref": wp.vec2(config.contact_time_constant_s, config.contact_damping_ratio),
                    "mujoco:geom_solimp": [config.contact_impedance, config.contact_impedance, 0.001, 0.5, 2.0],
                },
            )
        )
    # The wrist capsule and jaw bases approximate adjoining pieces of one
    # mechanism. Their round proxies overlap inside the wrist-yaw hinge.
    # An intervening geometry-free yaw link prevents the usual parent-child
    # exclusion from recognizing this adjacency. Exclude only those two
    # pairs; retain shaft/jaw, jaw/jaw, inter-tool and external contacts.
    for jaw_shape in shapes[2:]:
        builder.add_shape_collision_filter_pair(shapes[1], jaw_shape)
    return RCMGrasperAssembly(
        config=config,
        palm_body=bodies[5],
        jaw_bodies=(bodies[6], bodies[7]),
        palm_joint=joints[5],
        jaw_joints=tuple(jaw_joints),
        jaw_q_indices=tuple(jaw_q),
        jaw_dof_indices=tuple(jaw_dof),
        capsule_bodies=capsule_bodies,
        capsule_shapes=tuple(shapes),
        capsule_endpoints=endpoints,
        capsule_radii=radii,
        arm_joints=tuple(joints),
        arm_q_indices=tuple(q_indices),
        arm_dof_indices=tuple(dof_indices),
        arm_lower=lower,
        arm_upper=upper,
        nominal_arm_positions=np.asarray(_NOMINAL_ARM_POSITIONS, dtype=float),
        nominal_insertion_m=nominal_insertion_m,
        insertion_limits_m=tuple(float(limit) for limit in insertion_limits),
        part_body_indices=dict(zip(("roll", "pitch", "yaw", "jaw1", "jaw2"), bodies[3:], strict=True)),
        body_indices=tuple(bodies),
        trocar_pose=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=float),
    )
