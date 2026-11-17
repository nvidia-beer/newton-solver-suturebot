# SPDX-License-Identifier: Apache-2.0
"""Bounded native MuJoCo pose IK that writes only an independent scratch state.

Returned joint values are actuator targets. The caller must never copy the
scratch poses or coordinates into a running physical state. A fixed-base RCM
chain enforces its shaft constraint through its joints, not through this solver.
"""

from __future__ import annotations

import mujoco
import numpy as np


class UnreachablePoseError(ValueError):
    """A bounded joint solution did not meet the requested pose tolerances."""


def _pose(value):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (7,) or not np.isfinite(result).all() or np.linalg.norm(result[3:]) < 1e-8:
        raise ValueError("IK requires a finite xyz/xyzw target with a nonzero quaternion")
    result[3:] /= np.linalg.norm(result[3:])
    if result[6] < 0:
        result[3:] *= -1
    return result


def _rotation_error(target_xyzw, actual_matrix):
    actual, conjugate, difference = np.empty(4), np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(actual, actual_matrix)
    mujoco.mju_negQuat(conjugate, actual)
    mujoco.mju_mulQuat(difference, target_xyzw[[3, 0, 1, 2]], conjugate)
    if difference[0] < 0:
        difference *= -1
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, difference, 1.0)
    return result


def _right_log_jacobian_inverse(rotation):
    """Derivative of Log(R exp(delta)) with respect to a local delta."""
    x, y, z = rotation
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = float(np.linalg.norm(rotation))
    coefficient = 1 / 12 if angle < 1e-6 else (1 - 0.5 * angle / np.tan(0.5 * angle)) / angle**2
    return np.eye(3) + 0.5 * skew + coefficient * (skew @ skew)


class NativePoseIK:
    """Warm-started six-coordinate IK against the actual simulation model.

    ``joint_ids`` contains native MuJoCo scalar hinge/slide joint IDs, in the
    order expected by the caller's actuator targets. ``body_id`` is the native
    tool/palm body whose world pose is commanded. ``solve`` reads the live qpos
    array but changes only its own MjData. Jaw targets remain the caller's job.
    """

    def __init__(
        self,
        model,
        joint_ids,
        body_id,
        *,
        position_tolerance_m=1e-5,
        orientation_tolerance_rad=1e-3,
        rotation_length_m=0.01,
        max_iterations=100,
    ):
        ids = np.asarray(joint_ids, dtype=int)
        if ids.shape != (6,) or len(np.unique(ids)) != 6 or np.any(ids < 0) or np.any(ids >= model.njnt):
            raise ValueError("Pose IK requires six distinct valid native joint IDs")
        if not 0 < body_id < model.nbody:
            raise ValueError("Pose IK requires a non-world native tool body")
        if not np.all(
            np.isin(
                model.jnt_type[ids],
                [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE],
            )
        ):
            raise ValueError("Pose IK accepts only scalar hinge and slide coordinates")
        if not np.all(model.jnt_limited[ids]):
            raise ValueError("Every IK joint needs finite explicit limits")
        self.lower, self.upper = model.jnt_range[ids].T.copy()
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("IK joint limits must be finite and increasing")
        values = [position_tolerance_m, orientation_tolerance_rad, rotation_length_m]
        if not np.isfinite(values).all() or min(values) <= 0 or max_iterations < 1:
            raise ValueError("IK tolerances, rotation scale and iteration budget must be positive")
        self.model, self.joint_ids, self.body_id = model, ids.copy(), int(body_id)
        self.coordinates, self.dofs = (
            model.jnt_qposadr[ids].copy(),
            model.jnt_dofadr[ids].copy(),
        )
        ancestors, body = set(), self.body_id
        while body:
            ancestors.add(body)
            body = int(model.body_parentid[body])
        if not set(model.jnt_bodyid[ids]).issubset(ancestors):
            raise ValueError("Each IK joint must belong to the commanded tool's ancestor chain")
        ancestor_joints = [
            joint for joint in range(model.njnt) if model.jnt_bodyid[joint] in ancestors and joint not in ids
        ]
        widths = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
        self.external_coordinates = np.array(
            [
                coordinate
                for joint in ancestor_joints
                for coordinate in range(
                    model.jnt_qposadr[joint],
                    model.jnt_qposadr[joint] + widths.get(model.jnt_type[joint], 1),
                )
            ],
            dtype=int,
        )
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.rotation_length_m, self.max_iterations = (
            float(rotation_length_m),
            int(max_iterations),
        )
        self.scales = np.where(model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_SLIDE, 0.05, 1.0)
        self.scratch = mujoco.MjData(model)
        self.jacobian_position = np.empty((3, model.nv))
        self.jacobian_rotation = np.empty((3, model.nv))
        self.reset(model.qpos0)

    def reset(self, actual_qpos):
        """Forget warm starts/cache after a physical reset; write no live data."""
        values = self._base(actual_qpos)
        self.guess = np.clip(values[self.coordinates], self.lower, self.upper)
        self._target = self._external = self._solution = None
        self.last_diagnostics = {"converged": False, "cached": False, "iterations": 0}

    def _base(self, actual_qpos):
        values = np.asarray(actual_qpos, dtype=float)
        if values.shape != (self.model.nq,) or not np.isfinite(values).all():
            raise ValueError("IK needs the complete finite native qpos vector")
        return values.copy()

    def _evaluate(self, values, target):
        self.scratch.qpos[self.coordinates] = values
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        rotation = _rotation_error(target[3:], self.scratch.xmat[self.body_id])
        residual = np.r_[
            target[:3] - self.scratch.xpos[self.body_id],
            self.rotation_length_m * rotation,
        ]
        mujoco.mj_jacBody(
            self.model,
            self.scratch,
            self.jacobian_position,
            self.jacobian_rotation,
            self.body_id,
        )
        jacobian = np.vstack(
            (
                -self.jacobian_position[:, self.dofs],
                -self.rotation_length_m * _right_log_jacobian_inverse(rotation) @ self.jacobian_rotation[:, self.dofs],
            )
        )
        return residual, jacobian

    def solve(self, target_pose, actual_qpos, *, initial=None):
        """Return six bounded actuator targets, or fail without changing physics."""
        target, base = _pose(target_pose), self._base(actual_qpos)
        external = base[self.external_coordinates]
        if initial is None and np.array_equal(target, self._target) and np.array_equal(external, self._external):
            self.last_diagnostics = {
                **self.last_diagnostics,
                "cached": True,
                "iterations": 0,
            }
            return self._solution.copy()
        guess = self.guess if initial is None else np.asarray(initial, dtype=float)
        if guess.shape != (6,) or not np.isfinite(guess).all():
            raise ValueError("IK initialization needs six finite joint values")
        self.scratch.qpos[:] = base
        values = np.clip(guess, self.lower, self.upper)
        damping, iterations = 1e-8, 0
        for iteration in range(self.max_iterations):
            residual, jacobian = self._evaluate(values, target)
            if self._converged(residual):
                break
            iterations = iteration + 1
            scaled = jacobian * self.scales
            gradient = scaled.T @ residual
            # A descent direction pointing beyond a bound has no usable DOF.
            active = ((values <= self.lower + 1e-12) & (gradient > 0)) | (
                (values >= self.upper - 1e-12) & (gradient < 0)
            )
            scaled[:, active] = 0
            gradient = scaled.T @ residual
            normal, cost = scaled.T @ scaled, float(residual @ residual)
            accepted = False
            for _ in range(8):
                step = np.linalg.solve(normal + damping * np.eye(6), -gradient)
                # Bound a single kinematic iteration independently of the joint
                # limits: at most 0.3 rad or 15 mm per scaled coordinate.
                step *= min(1.0, 0.3 / max(float(np.linalg.norm(step)), 1e-12))
                for reduction in range(9):
                    fraction = 0.5**reduction
                    candidate = np.clip(values + fraction * self.scales * step, self.lower, self.upper)
                    trial, _ = self._evaluate(candidate, target)
                    if self._converged(trial) or float(trial @ trial) < cost * (1 - 1e-4 * fraction):
                        values = candidate
                        damping = max(1e-12, damping * 0.3)
                        accepted = True
                        break
                if accepted:
                    break
                damping *= 10
            if not accepted:
                break
        residual, _ = self._evaluate(values, target)
        position_error = float(np.linalg.norm(residual[:3]))
        orientation_error = float(np.linalg.norm(residual[3:]) / self.rotation_length_m)
        converged = position_error <= self.position_tolerance_m and orientation_error <= self.orientation_tolerance_rad
        self.last_diagnostics = {
            "converged": bool(converged),
            "cached": False,
            "iterations": iterations,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
        }
        if not converged:
            raise UnreachablePoseError(
                f"RCM tool pose is unreachable within joint limits/tolerances: "
                f"position error {position_error:.6g} m, orientation error {orientation_error:.6g} rad"
            )
        self.guess = values.copy()
        self._target, self._external, self._solution = (
            target.copy(),
            external.copy(),
            values.copy(),
        )
        return values.copy()

    def _converged(self, residual):
        return bool(
            np.linalg.norm(residual[:3]) <= self.position_tolerance_m
            and np.linalg.norm(residual[3:]) <= self.orientation_tolerance_rad * self.rotation_length_m
        )
