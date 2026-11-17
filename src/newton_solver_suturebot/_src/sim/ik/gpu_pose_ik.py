# SPDX-License-Identifier: Apache-2.0
"""Private fixed-base six-coordinate pose IK; outputs actuator targets only.

Same bounded damped least-squares, active bounds and line-search budgets as
NativePoseIK. Authored native body/joint frames define forward kinematics.
Double precision changes no geometric tolerance; unsupported chains fail closed.
"""

from __future__ import annotations

import threading

import numpy as np
import warp as wp

from newton_solver_suturebot._src.sim.ik.pose_ik import NativePoseIK, UnreachablePoseError, _pose
from newton_solver_suturebot._src.solvers.fem.differentiation import active_tape

V6 = wp.types.vector(length=6, dtype=wp.float64)
M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
M63 = wp.types.matrix(shape=(6, 3), dtype=wp.float64)


@wp.func
def evaluate(
    q: V6,
    target_p: wp.vec3d,
    target_q: wp.quatd,
    body_p: wp.array[wp.vec3d],
    body_q: wp.array[wp.quatd],
    joint_body: wp.array[int],
    joint_type: wp.array[int],
    joint_pos: wp.array[wp.vec3d],
    joint_axis: wp.array[wp.vec3d],
    joint_ref: wp.array[wp.float64],
    length: wp.float64,
):
    p = wp.vec3d(0.0)
    rot = wp.quatd(0.0, 0.0, 0.0, 1.0)
    anchors = M63(0.0)
    axes = M63(0.0)
    for body in range(body_p.shape[0]):
        p = p + wp.quat_rotate(rot, body_p[body])
        rot = rot * body_q[body]
        for j in range(6):
            if joint_body[j] == body:
                axis = wp.quat_rotate(rot, joint_axis[j])
                anchor = p + wp.quat_rotate(rot, joint_pos[j])
                for k in range(3):
                    axes[j, k] = axis[k]
                    anchors[j, k] = anchor[k]
                delta = q[j] - joint_ref[j]
                if joint_type[j] == 2:
                    p = p + axis * delta
                else:
                    rot = rot * wp.quat_from_axis_angle(joint_axis[j], delta)
                    p = anchor - wp.quat_rotate(rot, joint_pos[j])
    difference = target_q * wp.quat_inverse(rot)
    if difference[3] < wp.float64(0.0):
        difference = -difference
    xyz = wp.vec3d(difference[0], difference[1], difference[2])
    sine = wp.length(xyz)
    rotation = wp.vec3d(0.0)
    if sine > wp.float64(1e-15):
        rotation = xyz * (wp.float64(2.0) * wp.atan2(sine, difference[3]) / sine)
    error = V6(0.0)
    translation = target_p - p
    for k in range(3):
        error[k] = translation[k]
        error[k + 3] = length * rotation[k]
    angle = wp.length(rotation)
    coefficient = wp.float64(1.0 / 12.0)
    if angle >= wp.float64(1e-6):
        coefficient = (wp.float64(1.0) - wp.float64(0.5) * angle / wp.tan(wp.float64(0.5) * angle)) / (angle * angle)
    jacobian = M6(0.0)
    for j in range(6):
        axis = wp.vec3d(axes[j, 0], axes[j, 1], axes[j, 2])
        position = axis
        angular = wp.vec3d(0.0)
        if joint_type[j] == 3:
            anchor = wp.vec3d(anchors[j, 0], anchors[j, 1], anchors[j, 2])
            position = wp.cross(axis, p - anchor)
            angular = (
                axis
                + wp.float64(0.5) * wp.cross(rotation, axis)
                + coefficient * wp.cross(rotation, wp.cross(rotation, axis))
            )
        for k in range(3):
            jacobian[k, j] = -position[k]
            jacobian[k + 3, j] = -length * angular[k]
    return error, jacobian


@wp.func
def converged(error: V6, pos_tol: wp.float64, rot_tol_scaled: wp.float64):
    return (
        wp.length(wp.vec3d(error[0], error[1], error[2])) <= pos_tol
        and wp.length(wp.vec3d(error[3], error[4], error[5])) <= rot_tol_scaled
    )


@wp.func
def cholesky_solve(matrix: M6, rhs: V6):
    factor = M6(0.0)
    valid = wp.int32(1)
    for row in range(6):
        for column in range(row + 1):
            value = matrix[row, column]
            for k in range(column):
                value = value - factor[row, k] * factor[column, k]
            if row == column:
                if not wp.isfinite(value) or value <= wp.float64(0.0):
                    valid = 0
                    value = wp.float64(1.0)
                factor[row, column] = wp.sqrt(value)
            else:
                factor[row, column] = value / factor[column, column]
    intermediate = V6(0.0)
    result = V6(0.0)
    for row in range(6):
        value = rhs[row]
        for k in range(row):
            value = value - factor[row, k] * intermediate[k]
        intermediate[row] = value / factor[row, row]
    for reverse in range(6):
        row = 5 - reverse
        value = intermediate[row]
        for k in range(row + 1, 6):
            value = value - factor[k, row] * result[k]
        result[row] = value / factor[row, row]
        if not wp.isfinite(result[row]):
            valid = 0
    return result, valid


@wp.kernel(enable_backward=False)
def solve_kernel(
    body_p: wp.array[wp.vec3d],
    body_q: wp.array[wp.quatd],
    joint_body: wp.array[int],
    joint_type: wp.array[int],
    joint_pos: wp.array[wp.vec3d],
    joint_axis: wp.array[wp.vec3d],
    joint_ref: wp.array[wp.float64],
    lower: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    scales: wp.array[wp.float64],
    target: wp.array[wp.float64],
    guess: wp.array[wp.float64],
    accepted_guess: wp.array[wp.float64],
    result: wp.array[wp.float64],
    pos_tol: wp.float64,
    rot_tol: wp.float64,
    length: wp.float64,
    max_iterations: int,
):
    values = V6(0.0)
    for j in range(6):
        values[j] = wp.clamp(guess[j], lower[j], upper[j])
    target_p = wp.vec3d(target[0], target[1], target[2])
    target_q = wp.quatd(target[3], target[4], target[5], target[6])
    damping = wp.float64(1e-8)
    iterations = wp.int32(0)
    for iteration in range(max_iterations):
        error, jac = evaluate(
            values,
            target_p,
            target_q,
            body_p,
            body_q,
            joint_body,
            joint_type,
            joint_pos,
            joint_axis,
            joint_ref,
            length,
        )
        if converged(error, pos_tol, rot_tol * length):
            break
        iterations = iteration + 1
        scaled = M6(0.0)
        gradient = V6(0.0)
        for row in range(6):
            for column in range(6):
                scaled[row, column] = jac[row, column] * scales[column]
                gradient[column] += scaled[row, column] * error[row]
        for column in range(6):
            active = (values[column] <= lower[column] + wp.float64(1e-12) and gradient[column] > wp.float64(0.0)) or (
                values[column] >= upper[column] - wp.float64(1e-12) and gradient[column] < wp.float64(0.0)
            )
            if active:
                gradient[column] = wp.float64(0.0)
                for row in range(6):
                    scaled[row, column] = wp.float64(0.0)
        normal = M6(0.0)
        for row in range(6):
            for column in range(6):
                value = wp.float64(0.0)
                for k in range(6):
                    value += scaled[k, row] * scaled[k, column]
                normal[row, column] = value
        cost = wp.dot(error, error)
        accepted = wp.int32(0)
        for damp_attempt in range(8):  # noqa: B007 - fixed kernel iteration budget
            system = normal
            for j in range(6):
                system[j, j] += damping
            step, valid = cholesky_solve(system, -gradient)
            if valid == 0:
                damping = damping * wp.float64(10.0)
                continue
            step = step * wp.min(
                wp.float64(1.0),
                wp.float64(0.3) / wp.max(wp.length(step), wp.float64(1e-12)),
            )
            fraction = wp.float64(1.0)
            for reduction in range(9):  # noqa: B007 - fixed kernel line-search budget
                candidate = V6(0.0)
                for j in range(6):
                    candidate[j] = wp.clamp(values[j] + fraction * scales[j] * step[j], lower[j], upper[j])
                trial, _trial_jac = evaluate(
                    candidate,
                    target_p,
                    target_q,
                    body_p,
                    body_q,
                    joint_body,
                    joint_type,
                    joint_pos,
                    joint_axis,
                    joint_ref,
                    length,
                )
                if converged(trial, pos_tol, rot_tol * length) or wp.dot(trial, trial) < cost * (
                    wp.float64(1.0) - wp.float64(1e-4) * fraction
                ):
                    values = candidate
                    damping = wp.max(wp.float64(1e-12), damping * wp.float64(0.3))
                    accepted = 1
                    break
                fraction = fraction * wp.float64(0.5)
            if accepted != 0:
                break
            damping = damping * wp.float64(10.0)
        if accepted == 0:
            break
    error, jac = evaluate(
        values,
        target_p,
        target_q,
        body_p,
        body_q,
        joint_body,
        joint_type,
        joint_pos,
        joint_axis,
        joint_ref,
        length,
    )
    ok = converged(error, pos_tol, rot_tol * length)
    for j in range(6):
        result[j] = values[j]
        if ok:
            accepted_guess[j] = values[j]
    result[6] = wp.float64(ok)
    result[7] = wp.float64(iterations)
    result[8] = wp.length(wp.vec3d(error[0], error[1], error[2]))
    result[9] = wp.length(wp.vec3d(error[3], error[4], error[5])) / length


@wp.kernel(enable_backward=False)
def write_targets(
    accepted_guess: wp.array[wp.float64],
    indices: wp.array[int],
    targets: wp.array[wp.float32],
):
    j = wp.tid()
    targets[indices[j]] = wp.float32(accepted_guess[j])


@wp.kernel(enable_backward=False)
def evaluate_kernel(
    body_p: wp.array[wp.vec3d],
    body_q: wp.array[wp.quatd],
    joint_body: wp.array[int],
    joint_type: wp.array[int],
    joint_pos: wp.array[wp.vec3d],
    joint_axis: wp.array[wp.vec3d],
    joint_ref: wp.array[wp.float64],
    values: wp.array[wp.float64],
    target: wp.array[wp.float64],
    length: wp.float64,
    errors: wp.array[wp.float64],
    jacobians: wp.array2d[wp.float64],
):
    q = V6(0.0)
    for j in range(6):
        q[j] = values[j]
    residual, jac = evaluate(
        q,
        wp.vec3d(target[0], target[1], target[2]),
        wp.quatd(target[3], target[4], target[5], target[6]),
        body_p,
        body_q,
        joint_body,
        joint_type,
        joint_pos,
        joint_axis,
        joint_ref,
        length,
    )
    for row in range(6):
        errors[row] = residual[row]
        for column in range(6):
            jacobians[row, column] = jac[row, column]


class GPUPoseIK:
    """A private device workspace for one fixed-base arm. No physical state writes.

    The device arrays hold the native chain as it is at construction. Every reset and every
    solve that runs the kernel compares the native chain with that snapshot first and raises if
    it differs. A request for the target of the last converged solve returns that solution from
    the host without the comparison: a chain edited after that solve is reported by the next new
    target, initial guess or reset. The thread, stream and eager-execution checks run on every call.
    """

    def __init__(self, model, joint_ids, body_id, *, device=None, **options):
        self.reference = NativePoseIK(model, joint_ids, body_id, **options)
        if self.reference.external_coordinates.size:
            raise ValueError("GPU IK supports only fixed-base chains with these six scalar joints")
        self.position_tolerance_m = self.reference.position_tolerance_m
        self.orientation_tolerance_rad = self.reference.orientation_tolerance_rad
        self.model = model
        self.device = wp.get_device(device)
        self.thread = threading.current_thread()
        self.stream = wp.get_stream(self.device) if self.device.is_cuda else None
        bodies = []
        body = int(body_id)
        while body:
            bodies.append(body)
            body = int(model.body_parentid[body])
        self.bodies = np.array(bodies[::-1], dtype=int)
        if np.any(model.body_mocapid[self.bodies] >= 0):
            raise ValueError("GPU IK excludes moving mocap ancestors")
        self.ids = self.reference.joint_ids
        # Body traversal is topological; native joints within a body are applied in native order.
        if list(self.ids) != sorted(self.ids):
            raise ValueError("GPU IK currently requires native joint ordering")
        self.snapshot = self._model_snapshot()
        self.arrays = (
            wp.array(model.body_pos[self.bodies], dtype=wp.vec3d, device=self.device),
            wp.array(
                model.body_quat[self.bodies][:, [1, 2, 3, 0]],
                dtype=wp.quatd,
                device=self.device,
            ),
            wp.array(
                [list(self.bodies).index(int(model.jnt_bodyid[j])) for j in self.ids],
                dtype=int,
                device=self.device,
            ),
            wp.array(model.jnt_type[self.ids], dtype=int, device=self.device),
            wp.array(model.jnt_pos[self.ids], dtype=wp.vec3d, device=self.device),
            wp.array(model.jnt_axis[self.ids], dtype=wp.vec3d, device=self.device),
            wp.array(
                model.qpos0[self.reference.coordinates],
                dtype=wp.float64,
                device=self.device,
            ),
        )
        self.lower = wp.array(self.reference.lower, dtype=wp.float64, device=self.device)
        self.upper = wp.array(self.reference.upper, dtype=wp.float64, device=self.device)
        self.scales = wp.array(self.reference.scales, dtype=wp.float64, device=self.device)
        self.guess = wp.array(self.reference.guess, dtype=wp.float64, device=self.device)
        self.initial = wp.empty(6, dtype=wp.float64, device=self.device)
        self.target = wp.empty(7, dtype=wp.float64, device=self.device)
        self.result = wp.zeros(10, dtype=wp.float64, device=self.device)
        self._target = None
        self._solution = None
        self._indices = {}
        self.last_diagnostics = {
            "converged": False,
            "cached": False,
            "iterations": 0,
            "backend": "warp-private-kinematics",
        }

    def _model_snapshot(self):
        m, ids, bodies = self.model, self.ids, self.bodies
        return tuple(
            np.asarray(value).copy()
            for value in (
                m.body_pos[bodies],
                m.body_quat[bodies],
                m.body_parentid[bodies],
                m.body_mocapid[bodies],
                m.jnt_type[ids],
                m.jnt_bodyid[ids],
                m.jnt_pos[ids],
                m.jnt_axis[ids],
                m.jnt_range[ids],
                m.jnt_limited[ids],
                m.jnt_qposadr[ids],
                m.qpos0[self.reference.coordinates],
            )
        )

    def _validate(self):
        if threading.current_thread() is not self.thread:
            raise RuntimeError("GPU IK workspace belongs to another thread")
        if self.device.is_cuda and wp.get_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("GPU IK requires its original ordered stream")
        if self.device.is_capturing or active_tape() is not None:
            raise ValueError("GPU IK host convergence guard requires non-differentiable eager execution")

    def _validate_model(self):
        if not all(np.array_equal(a, b) for a, b in zip(self.snapshot, self._model_snapshot(), strict=True)):
            raise ValueError("GPU IK model changed; rebuild the private descriptor")

    def reset(self, actual_qpos):
        self._validate()
        self._validate_model()
        self.reference.reset(actual_qpos)
        self.guess.assign(self.reference.guess)
        self._target = self._solution = None
        self.last_diagnostics = {
            "converged": False,
            "cached": False,
            "iterations": 0,
            "backend": "warp-private-kinematics",
        }

    def solve(self, target_pose, actual_qpos=None, *, initial=None):
        self._validate()
        try:
            target = _pose(target_pose)
            if actual_qpos is not None:
                self.reference._base(actual_qpos)  # API compatibility; fixed-base chain needs no live coordinates.
        except Exception:
            # An invalid request reports a changed model first, as a request that is solved does.
            self._validate_model()
            raise
        if initial is None and np.array_equal(target, self._target):
            # The cached solution was computed with the chain the device arrays hold; returning
            # it touches neither the device nor the native model, so the chain is not compared.
            self.last_diagnostics = {
                **self._successful_diagnostics,
                "cached": True,
                "iterations": 0,
            }
            return self._solution.copy()
        self._validate_model()
        if initial is not None:
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (6,) or not np.isfinite(initial).all():
                raise ValueError("IK initialization needs six finite joint values")
            self.initial.assign(np.clip(initial, self.reference.lower, self.reference.upper))
        self.target.assign(target)
        r = self.reference
        wp.launch(
            solve_kernel,
            1,
            inputs=[
                *self.arrays,
                self.lower,
                self.upper,
                self.scales,
                self.target,
                self.guess if initial is None else self.initial,
                self.guess,
                self.result,
                r.position_tolerance_m,
                r.orientation_tolerance_rad,
                r.rotation_length_m,
                r.max_iterations,
            ],
            device=self.device,
            block_dim=1,
        )
        packet = self.result.numpy()
        self.last_diagnostics = {
            "converged": bool(packet[6]),
            "cached": False,
            "iterations": int(packet[7]),
            "position_error_m": float(packet[8]),
            "orientation_error_rad": float(packet[9]),
            "backend": "warp-private-kinematics",
        }
        if not packet[6] or not np.isfinite(packet).all():
            raise UnreachablePoseError("RCM GPU tool pose is unreachable within unchanged joint limits/tolerances")
        self._target = target.copy()
        self._successful_diagnostics = self.last_diagnostics.copy()
        self._solution = packet[:6].copy()
        return self._solution.copy()

    def solve_into(self, target_pose, targets, indices, *, initial=None):
        if (
            targets.dtype != wp.float32
            or targets.requires_grad
            or targets.grad is not None
            or targets.device != self.device
            or not targets.is_contiguous
        ):
            raise ValueError("GPU IK targets require matching contiguous non-gradient device storage")
        key = tuple(int(i) for i in indices)
        if len(key) != 6 or len(set(key)) != 6 or min(key) < 0 or max(key) >= targets.size:
            raise ValueError("GPU IK needs six distinct actuator target indices")
        self.solve(target_pose, initial=initial)
        if key not in self._indices:
            self._indices[key] = wp.array(key, dtype=int, device=self.device)
        wp.launch(
            write_targets,
            6,
            [self.guess, self._indices[key], targets],
            device=self.device,
        )
