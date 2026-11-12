"""Procedural needle-arc commands through the contact-driven distal gripper.

The source example supplies the desired needle arc. A settled, measured grasp
frame converts that arc to gripper targets. The scene solves bounded joint IK
for the fixed-trocar tools; finite actuators and contact determine actual motion.
No policy or needle pose drive is used.
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


def action_to(old, target, config, *, strict=False):
    """Bound a world-frame gripper pose increment by the configured rate limits."""
    rotation = np.asarray(wp.quat(*target[3:]) * wp.quat_inverse(wp.quat(*old[3:])), dtype=float)
    if rotation[3] < 0:
        rotation = -rotation
    sine = np.linalg.norm(rotation[:3])
    rotvec = rotation[:3] * 2 * np.arctan2(sine, rotation[3]) / max(sine, 1e-12)
    raw = np.r_[(target[:3] - old[:3]) / config.translation_step, rotvec / config.rotation_step]
    if strict and np.max(np.abs(raw)) > 1 + 1e-5:
        raise ValueError("Reference command exceeds the configured action-rate limits")
    return np.clip(raw, -1, 1)


def ramp_distance(time, speed, ramp_time=0.5):
    """Integrated smoothstep speed ramp: zero initial speed and acceleration."""
    u = np.clip(time / ramp_time, 0.0, 1.0)
    return speed * (ramp_time * (u**3 - 0.5 * u**4) + np.maximum(0.0, time - ramp_time))


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


class ProceduralGraspController:
    """Follow one source entry/exit arc using a fixed measured grasp frame.

    Each world settles independently before starting. Loss of bilateral jaw
    contact pauses its commands; 0.25 seconds of continuous loss stops it until
    reset. The grasp frame stays fixed even if the needle slips. Completion of
    the commanded arc is reported separately from actual entry and exit events.

    ``geometry`` contains ``control_mode="physical_grasp"`` and the scene's
    ``insertion_plan``. ``config`` supplies that control mode and positive
    ``frame_dt``, ``translation_step``, ``rotation_step`` and ``jaw_step_rad``.
    Calling the controller with observation dictionaries returns one float32
    seven-value action per observation: world translation, world rotation and
    jaw closure. Apply these to gripper actuator targets, never to needle state.
    ``env_id`` and ``reset_count`` keep reset and contact history independent.
    """

    def __init__(self, geometry, config, speed=0.003, settle_seconds=0.5):
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

    def __call__(self, snapshots):
        actions = np.zeros((len(snapshots), 7), dtype=np.float32)
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
                state.status, state.reason = "stopped", str(failure or "Physics reported failure")
            if state.status == "stopped":
                continue

            relative, old_target = _pose(item["needle_relative_pose"]), _pose(item["gripper_target_pose"])
            counts = np.asarray(item["jaw_contact_count"], dtype=float)
            forces = np.asarray(item["jaw_contact_normal_force_n"], dtype=float)
            jaw_target = np.asarray(item["jaw_target_rad"], dtype=float)
            if any(value.shape != (2,) or not np.isfinite(value).all() for value in (counts, forces, jaw_target)):
                raise ValueError("Procedural control needs two finite jaw counts, forces and targets")
            contact = bool(np.all(counts > 0) and np.all(forces > 1e-5))
            if state.grasp is None:
                actions[index, 6] = np.clip(-(jaw_target[0] - jaw_target[1]) / self.config.jaw_step_rad, -1, 0)
                if now - state.started < self.settle_seconds:
                    continue
                if not contact:
                    state.status = "waiting_for_grasp"
                    if now - state.started >= self.settle_seconds + 2.0:
                        state.status, state.reason = "stopped", "Bilateral needle grasp was not established"
                        actions[index] = 0
                    continue
                state.grasp = relative.copy()
                state.had_contact = True
                elapsed = 0.0

            state.relative_translation_m = float(np.linalg.norm(relative[:3] - state.grasp[:3]))
            similarity = np.clip(abs(np.dot(relative[3:], state.grasp[3:])), 0, 1)
            state.relative_rotation_rad = float(2 * np.arccos(similarity))
            if not contact:
                if state.lost_since is None:
                    state.lost_since = now
                state.status = "grasp_lost"
                state.had_contact = False
                if now - state.lost_since >= 0.25 - 1e-9:
                    state.status, state.reason = "stopped", "Bilateral needle contact lost for 0.25 seconds"
                continue
            if state.had_contact:
                state.progress_time += elapsed
            state.had_contact, state.lost_since = True, None
            state.angle = min(
                self.initial_angle + float(ramp_distance(state.progress_time, self.speed)) / self.radius,
                self.finish_angle,
            )
            state.status = "holding" if state.angle >= self.finish_angle else "inserting"
            target = gripper_target(pose_at(self.plan, state.angle), state.grasp)
            actions[index, :6] = action_to(old_target, target, self.config)
            actions[index, 6] = np.clip(-(jaw_target[0] - jaw_target[1]) / self.config.jaw_step_rad, -1, 0)
        return actions

    def report(self):
        """Plain diagnostics; a completed target arc is not a success claim."""
        return [
            {
                "env_id": env_id,
                "phase": state.status,
                "reason": state.reason,
                "commanded_angle_rad": state.angle,
                "commanded_arc_complete": state.angle >= self.finish_angle,
                "entry_count": state.entries,
                "exit_count": state.exits,
                "relative_translation_m": state.relative_translation_m,
                "relative_rotation_rad": state.relative_rotation_rad,
            }
            for env_id, state in sorted(self.scenes.items())
        ]
