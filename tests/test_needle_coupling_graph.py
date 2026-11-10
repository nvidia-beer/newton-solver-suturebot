# SPDX-License-Identifier: Apache-2.0
"""Captured trials preserve the public coupled-step transaction and fallback."""

import os
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import history, setup
from test_needle_tissue_coupler import _native_state

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge, NeedleTissueCoupler
from newton_solver_suturebot._src.tunnel_alm import _TunnelALM

_FIELDS = ("particle_q", "particle_qd", "particle_f", "joint_q", "joint_qd", "body_q", "body_qd", "body_f")
_DT = 0.0005


def _configure(example, mode, capture):
    # Accepted geometric probes seed a preloaded shaft. The subsequent tests
    # advance actual FEM/native rigid dynamics; the setup is not an insertion run.
    setup(example, _DT, 1500.0)
    example.interaction.friction = 0.02 if mode == "alm" else 0.2
    if mode == "alm":
        example.interaction.shaft_damping_per_m = 0
        example.interaction.tunnel_alm = _TunnelALM(
            example.interaction,
            penalty_per_m=1500,
            compliance_per_m=1 / 1500,
            clearance=2e-5,
            tolerance=2e-6,
        )
        example.interaction.tunnel_alm._resize()
    example.coupler = NeedleTissueCoupler(
        example.model,
        example.solver,
        example.rigid_solver,
        example.interaction,
        example.needle_body,
        example.drive.com,
        absolute_tolerance=2e-7,
        relative_tolerance=1e-4,
        max_iterations=160,
        shaft_preconditioner=True,
        capture_fem=True,
        capture_coupling=capture,
    )
    example.control.joint_f.assign(np.array([0, 1e-5, 0, 2e-8, 0, 0], np.float32))


@contextmanager
def _pair(mode):
    with ExitStack() as stack:
        reference = stack.enter_context(_example())
        captured = stack.enter_context(_example())
        _configure(reference, mode, False)
        _configure(captured, mode, True)
        yield reference, captured


def _arrays(state):
    return {name: getattr(state, name).numpy().copy() for name in _FIELDS}


def _snapshot(example):
    interaction = example.interaction
    result = {
        "input": _arrays(example.state_0),
        "output": _arrays(example.state_1),
        "native": _native_state(example.rigid_solver),
        "increment": example.solver.velocity_increment.numpy().copy(),
        "load": example.coupler._load.numpy().copy(),
        "history": history(interaction),
        "visited": tuple(sorted(interaction._visited_tets)),
        "openings": repr(interaction._openings),
    }
    for name in ("_previous_tip", "_previous_q", "_previous_samples"):
        result[name] = getattr(interaction, name).copy()
    for name in ("_track_nodes", "_track_bary", "_track_lengths", "_track_engaged"):
        result[name] = getattr(interaction, name).numpy().copy()
    if interaction.tunnel_alm is not None:
        result["normal"] = interaction.tunnel_alm.committed_normal.numpy().copy()
        result["tangent"] = interaction.tunnel_alm.committed_tangent.numpy().copy()
    return result


class NeedleCouplingGraphTests(unittest.TestCase):
    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda:
            self.skipTest("Coupled conditional-graph transaction requires CUDA")

    def assert_rollback(self, example, saved):
        current = _snapshot(example)
        for group in ("input", "output"):
            for name, values in saved[group].items():
                np.testing.assert_array_equal(current[group][name], values, err_msg=f"rollback {group}.{name}")
        for name in ("history", "visited", "openings"):
            self.assertEqual(current[name], saved[name], name)
        for name in saved.keys() - {"input", "output", "history", "visited", "openings"}:
            np.testing.assert_array_equal(current[name], saved[name], err_msg=f"rollback {name}")
        if example.interaction.tunnel_alm is not None:
            self.assertFalse(example.interaction.tunnel_alm._open)

    def assert_equivalent(self, reference, captured):
        self.assertEqual(history(reference.interaction), history(captured.interaction))
        for name in _FIELDS:
            expected = getattr(reference.state_1, name).numpy()
            actual = getattr(captured.state_1, name).numpy()
            self.assertTrue(np.isfinite(actual).all(), name)
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-8, err_msg=name)
        np.testing.assert_array_equal(
            captured.state_1.particle_q.numpy()[captured.fixed_vertex_indices],
            captured.state_0.particle_q.numpy()[captured.fixed_vertex_indices],
        )
        np.testing.assert_allclose(
            _native_state(captured.rigid_solver),
            _native_state(reference.rigid_solver),
            rtol=3e-6,
            atol=2e-7,
        )
        expected, actual = reference.coupler._value.numpy(), captured.coupler._value.numpy()
        self.assertLessEqual(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-20), 0.01)
        self.assertGreater(np.linalg.norm(expected), 1e-7)
        for example in (reference, captured):
            if example.interaction.tunnel_alm is not None:
                self.assertFalse(example.interaction.tunnel_alm._open)
        self.assertEqual(reference.coupler.coupling_graph_launches, 0)

    def step_pair(self, reference, captured, dt):
        for example in (reference, captured):
            native_time = float(example.rigid_solver.mj_data.time)
            with patch.object(example.interaction, "accept", wraps=example.interaction.accept) as accept:
                example.coupler.step(example.state_0, example.state_1, example.control, dt)
            self.assertEqual(accept.call_count, 1)
            self.assertAlmostEqual(example.rigid_solver.mj_data.time, native_time + dt, places=12)
        self.assert_equivalent(reference, captured)

    def test_penalty_and_alm_advances_match_with_alternating_buffers_timestep_and_loads(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                workspaces = None
                for index, dt in enumerate((_DT, _DT, _DT / 2)):
                    for example in (reference, captured):
                        example.drive.prepare_timestep(dt)
                        example.solver.refresh_tangent(example.state_0.particle_q, dt)
                        example.control.joint_f.assign(np.array([0, (-1) ** index * 1e-5, 0, 2e-8, 0, 0], np.float32))
                    self.step_pair(reference, captured, dt)
                    self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
                    current = tuple(id(workspace) for _, workspace in captured.coupler._captured_trials.linear_solvers)
                    if workspaces is not None:
                        self.assertEqual(current, workspaces)
                    workspaces = current
                    for example in (reference, captured):
                        example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertEqual(captured.coupler.coupling_graph_launches, 3)
                self.assertGreaterEqual(captured.coupler.coupling_graph_captures, 3)

    def test_exhausted_budget_rolls_back_then_retries_without_committing_history(self):
        for mode in ("penalty", "alm"):
            with self.subTest(mode=mode), _pair(mode) as (reference, captured):
                saved = _snapshot(captured)
                captured.coupler.max_iterations = 1
                with patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept:
                    with self.assertRaises(NeedleCouplingNotConverged):
                        captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                    self.assertEqual(accept.call_count, 0)
                self.assert_rollback(captured, saved)
                self.assertEqual(captured.coupler.coupling_graph_launches, 1)
                captured.coupler.max_iterations = 160
                self.step_pair(reference, captured, _DT)
                self.assertEqual(captured.coupler.coupling_graph_launches, 2)

    def test_late_resolution_rejection_restores_native_warmstart_output_and_alm_duals(self):
        with _pair("alm") as (reference, captured):
            saved = _snapshot(captured)
            with (
                patch.object(
                    captured.interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late guard")
                ),
                patch.object(captured.interaction, "accept", wraps=captured.interaction.accept) as accept,
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late guard"):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assertEqual(accept.call_count, 0)
            self.assert_rollback(captured, saved)
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)
            self.step_pair(reference, captured, _DT)
            self.assertEqual(captured.coupler.coupling_graph_launches, 2)
            self.assertEqual(captured.coupler.coupling_graph_captures, 1)

    def test_native_damping_change_at_fixed_dt_refreshes_capture(self):
        with _pair("penalty") as (reference, captured):
            self.step_pair(reference, captured, _DT)
            captures = captured.coupler.coupling_graph_captures
            for example in (reference, captured):
                # Preserve input allocation: only damping, not a different
                # input pointer or timestep, can invalidate this cached graph.
                example.state_0.assign(example.state_1)
                example.rigid_solver.mj_model.dof_damping[:] *= 1.5
            self.step_pair(reference, captured, _DT)
            self.assertGreater(captured.coupler.coupling_graph_captures, captures)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)

    def test_failed_preparation_is_not_cached_and_identical_retry_succeeds(self):
        with _pair("alm") as (reference, captured):
            saved = _snapshot(captured)
            original = captured.solver._make_linear_solver
            calls = [0]

            def once(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("injected preparation failure")
                return original(*args, **kwargs)

            with patch.object(captured.solver, "_make_linear_solver", side_effect=once):
                with self.assertRaisesRegex(RuntimeError, "injected preparation failure"):
                    captured.coupler.step(captured.state_0, captured.state_1, captured.control, _DT)
                self.assert_rollback(captured, saved)
                self.assertEqual(captured.coupler.coupling_graph_launches, 0)
                self.step_pair(reference, captured, _DT)
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)

    def test_near_exit_geometry_uses_original_authorization_and_commits_once(self):
        with _pair("penalty") as (reference, captured):
            for example in (reference, captured):
                middle = example.plan.exit_angle / 2
                poses = []
                for lift in np.linspace(0.0002, 0, 9)[1:]:
                    pose = example.plan.pose(middle)
                    pose[2] += lift
                    poses.append(pose)
                finish = example.plan.exit_angle + 0.01
                count = int(np.ceil((finish - middle) * example.geometry.radius / 0.00005))
                poses.extend(example.plan.pose(angle) for angle in np.linspace(middle, finish, count + 1)[1:])
                for pose in poses:
                    example.state_0.clear_forces()
                    example._external_forces.zero_()
                    example.interaction.add_forces(
                        example.state_0,
                        example.needle_body,
                        example.drive.com,
                        example._external_forces,
                        _DT,
                    )
                    example._initialize_pose(pose)
                    example.interaction.accept(example.state_0, example.needle_body)
                    self.assertNotEqual(example.interaction.status, "failed", example.interaction.reason)
                self.assertGreater(example.interaction.track_count, 0)
                self.assertFalse(example.interaction._in_material)
                example.state_0.clear_forces()
            self.step_pair(reference, captured, _DT)
            self.assertEqual(captured.coupler.coupling_graph_launches, 1)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 1)
            self.assertIn("surface-event fallback", captured.coupler.coupling_graph_status)


if __name__ == "__main__":
    unittest.main()
