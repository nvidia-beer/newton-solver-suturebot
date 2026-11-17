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

from ._dvrk_grasper import GrasperAssembly, GrasperConfig, _jaw_angles, _pose, add_tool_capsules, tool_instrument

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


def arm_link_poses(trocar_pose, palm_pose):
    """Poses of the roll link (shaft) and of the wrist-pitch link for a palm pose of the fixed-trocar tool.

    The inverse of :func:`_arm_pose` for the two links between the trocar and the palm, in closed form
    from the same joint frames. The origin of the wrist-pitch link is the wrist joint ``W``; the palm
    origin lies ``WRIST_LENGTH_M`` from it along the link's X axis, and the link's Z axis (the wrist-pitch
    axis) and X axis lie in the plane at right angles to the palm's Z axis (the wrist-yaw axis): in the
    palm frame ``X = (sin y, cos y, 0)`` and ``Z = (cos y, -sin y, 0)`` with ``y`` the wrist-yaw angle. The
    shaft runs from the trocar to ``W`` at right angles to the wrist-pitch axis, which gives
    ``tan y = d_x / d_y`` with ``d`` the offset from the trocar to the palm origin in the palm frame; of
    the two solutions the one within a quarter turn is the tool's (the joint is limited to less). The
    roll link has its origin at ``W``, its Z axis along the shaft away from the trocar and its Y axis
    along the wrist-pitch axis.

    Args:
        trocar_pose: Fixed pose of the trocar, xyz/xyzw; only its position is used.
        palm_pose: Pose of the palm (the wrist-yaw link), xyz/xyzw.

    Returns:
        The poses of the roll link and of the wrist-pitch link, xyz/xyzw.
    """
    trocar, palm = _pose(trocar_pose), _pose(palm_pose)
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*palm[3:])), dtype=float).reshape(3, 3)
    offset = rotation.T @ (palm[:3] - trocar[:3])
    yaw = float(np.arctan2(offset[0], offset[1])) if np.hypot(offset[0], offset[1]) > 0.0 else 0.0
    if yaw > 0.5 * np.pi:
        yaw -= np.pi
    elif yaw < -0.5 * np.pi:
        yaw += np.pi
    sine, cosine = np.sin(yaw), np.cos(yaw)
    link_x = rotation @ np.array([sine, cosine, 0.0])
    pitch_axis = rotation @ np.array([cosine, -sine, 0.0])
    wrist = palm[:3] - WRIST_LENGTH_M * link_x
    shaft = wrist - trocar[:3]
    shaft_z = shaft / np.linalg.norm(shaft)
    shaft_y = pitch_axis - (pitch_axis @ shaft_z) * shaft_z
    shaft_y /= np.linalg.norm(shaft_y)

    def pose(x, y, z):
        quaternion = np.asarray(wp.quat_from_matrix(wp.mat33(*np.column_stack((x, y, z)).reshape(-1))), dtype=float)
        return np.r_[wrist, quaternion / np.linalg.norm(quaternion)]

    return pose(np.cross(shaft_y, shaft_z), shaft_y, shaft_z), pose(link_x, np.cross(pitch_axis, link_x), pitch_axis)


def _rotation_matrices(quaternions):
    """Rotation matrices (..., 3, 3) of unit quaternions xyzw (..., 4)."""
    x, y, z, w = np.moveaxis(np.asarray(quaternions, dtype=float), -1, 0)
    return np.stack(
        (
            np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), axis=-1),
            np.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), axis=-1),
            np.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def arm_joint_positions(trocar_pose, palm_poses, nominal_insertion_m=NOMINAL_INSERTION_M):
    """Joint coordinates of the fixed-trocar tool for palm poses: the inverse of :func:`_arm_pose`, in closed form.

    The same joint frames as :func:`arm_link_poses`, for any number of palm poses at once. With ``y`` the
    wrist-yaw angle of that function, ``W`` the wrist joint and ``C`` the trocar:

    - the shaft ``W - C`` has the length ``nominal_insertion_m + insertion`` and, in the frame of the
      trocar, the direction ``(sin a cos p, -sin p, cos a cos p)`` with ``a`` the yaw and ``p`` the pitch
      about the trocar;
    - the wrist-pitch axis is the Y axis of the roll link, which the roll joint turns about the shaft
      from the Y axis the two trocar joints leave it: ``(-sin r, cos r, 0)`` in that frame;
    - the wrist link's X axis is ``cos w Z + sin w X`` of the roll link, with ``w`` the wrist pitch.

    A palm pose has one such set of coordinates with the wrist yaw within a quarter turn, the roll in
    (-pi, pi] and the trocar behind the wrist; it is returned whether or not it lies within the joint
    limits, which the caller compares it with. The scene's bounded pose IK reaches a palm pose exactly
    when this set lies within the limits (within its tolerances).

    Args:
        trocar_pose: Fixed pose of the trocar, xyz/xyzw.
        palm_poses: Poses of the palm (the wrist-yaw link), xyz/xyzw, shape (7,) or (count, 7).
        nominal_insertion_m: Distance of the wrist joint from the trocar at insertion zero [m].

    Returns:
        Yaw and pitch about the trocar [rad], insertion [m], roll, wrist pitch and wrist yaw [rad], in
        the order of ``ARM_JOINT_NAMES``; shape (6,) or (count, 6).
    """
    trocar = _pose(trocar_pose)
    palms = np.asarray(palm_poses, dtype=float)
    single = palms.ndim == 1
    palms = np.atleast_2d(palms)
    if palms.shape[1] != 7 or not np.isfinite(palms).all() or np.linalg.norm(palms[:, 3:], axis=1).min() < 1e-8:
        raise ValueError("Palm poses must be finite xyz/xyzw with nonzero quaternions")
    rotation = _rotation_matrices(palms[:, 3:] / np.linalg.norm(palms[:, 3:], axis=1, keepdims=True))
    base = _rotation_matrices(trocar[3:])
    offset = np.einsum("nji,nj->ni", rotation, palms[:, :3] - trocar[:3])
    wrist_yaw = np.arctan2(offset[:, 0], offset[:, 1])
    wrist_yaw = wrist_yaw - np.pi * (wrist_yaw > 0.5 * np.pi) + np.pi * (wrist_yaw < -0.5 * np.pi)
    sine, cosine = np.sin(wrist_yaw)[:, None], np.cos(wrist_yaw)[:, None]
    link_x = sine * rotation[:, :, 0] + cosine * rotation[:, :, 1]
    pitch_axis = cosine * rotation[:, :, 0] - sine * rotation[:, :, 1]
    shaft = palms[:, :3] - WRIST_LENGTH_M * link_x - trocar[:3]
    length = np.linalg.norm(shaft, axis=1)
    shaft_z = shaft / length[:, None]
    # The shaft and the wrist-pitch axis in the frame of the trocar.
    direction, axis = shaft_z @ base, pitch_axis @ base
    pitch = -np.arcsin(np.clip(direction[:, 1], -1.0, 1.0))
    yaw = np.arctan2(direction[:, 0], direction[:, 2])
    # The X and Y axes that the two trocar joints leave the roll link, in the frame of the trocar.
    before_x = np.stack((np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)), axis=1)
    before_y = np.stack((np.sin(yaw) * np.sin(pitch), np.cos(pitch), np.cos(yaw) * np.sin(pitch)), axis=1)
    roll = np.arctan2(-np.einsum("ni,ni->n", axis, before_x), np.einsum("ni,ni->n", axis, before_y))
    shaft_x = np.cross(pitch_axis, shaft_z)
    wrist_pitch = np.arctan2(np.einsum("ni,ni->n", link_x, shaft_x), np.einsum("ni,ni->n", link_x, shaft_z))
    result = np.stack((yaw, pitch, length - float(nominal_insertion_m), roll, wrist_pitch, wrist_yaw), axis=1)
    return result[0] if single else result


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
        ``opening_rad`` is the total opening of the jaws or one opening angle per
        jaw, as for the free-palm tool.
        """
        pose = None if palm_pose is None else _pose(palm_pose)
        if not self._anchor_initialized:
            raise RuntimeError("Place the fixed trocar before initializing its joint coordinates")
        jaw_angles = _jaw_angles(self.config, opening_rad)
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
        coordinates[list(self.jaw_q_indices)] = jaw_angles
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
    builder,
    *,
    config=None,
    label="grasper",
    nominal_insertion_m=NOMINAL_INSERTION_M,
    insertion_limits_m=(-0.03, 0.03),
    instrument=None,
):
    """Add two RCM hinges, axial slide, roll, two wrists and two jaws.

    The first hinge is attached to world, with no free palm or Cartesian pose
    constraint. The 80 mm nominal slide distance and +/-30 mm travel keep the
    fixed pivot inside the default 120 mm shaft. Custom insertion geometry
    must keep the pivot on the physical shaft throughout its travel. Dynamics
    and collision geometry share actual link frames.
    A -0.7 rad nominal wrist pitch leaves room for the insertion arc while
    preserving the original wrist limits; the fixed base follows exact FK.
    The collision capsules are the capsule set of the instrument, fitted to
    its drawn CAD with the shaft of the configured centreline length
    (``dvrk_tool_capsules``), on the roll, wrist-pitch and jaw links.
    ``instrument`` is "PSM1" or "PSM2"; by default the label if it is one of
    them, else "PSM1".
    """
    from ._dvrk_capsules import dvrk_tool_capsules

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
    capsules = add_tool_capsules(
        builder,
        dvrk_tool_capsules(tool_instrument(label, instrument), float(config.shaft_length_m)),
        {"roll": bodies[3], "pitch": bodies[4], "jaw1": bodies[6], "jaw2": bodies[7]},
        config,
        label,
    )
    return RCMGrasperAssembly(
        config=config,
        palm_body=bodies[5],
        jaw_bodies=(bodies[6], bodies[7]),
        palm_joint=joints[5],
        jaw_joints=tuple(jaw_joints),
        jaw_q_indices=tuple(jaw_q),
        jaw_dof_indices=tuple(jaw_dof),
        capsule_bodies=capsules[0],
        capsule_shapes=capsules[1],
        capsule_endpoints=capsules[2],
        capsule_radii=capsules[3],
        capsule_parts=capsules[4],
        capsule_bounds=capsules[5],
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
