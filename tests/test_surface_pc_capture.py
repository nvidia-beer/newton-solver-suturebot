"""Actual captured native surface forces, PC aliases and rejection contracts.

The real native/FEM contact fixture first seeds its initial accepted contact
metadata using the original force/accept API, without advancing physics. This
supplies the previous tip required by the conservative force-graph certificate.
Every CUDA test asserts graph activity; an eager fallback cannot pass silently.
"""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_tissue_coupler import _native_state
from test_surface_anderson_coupling import _contact, _snapshot
from test_surface_contact_preconditioner import enable

from newton_solver_suturebot._src import surface_preconditioner
from newton_solver_suturebot._src.coupling import (
    NeedleCouplingNotConverged,
    NeedleTissueCoupler,
)
from newton_solver_suturebot._src.host_trial_forces import CapturedHostTrialForces
from newton_solver_suturebot._src.insertion import NeedleStepTooLarge
from newton_solver_suturebot._src.needle_trial_forces import _NeedleTrialForces


def seed_contact(example, coupler):
    """Establish initial accepted geometry through the exact interaction API."""
    body_force = wp.clone(example.state_0.body_f)
    scratch = wp.zeros_like(coupler._forces)
    example.state_0.body_f.zero_()
    alm = example.interaction.tunnel_alm
    if alm is not None:
        alm.begin_step()
    try:
        example.interaction.add_forces(
            example.state_0,
            coupler.needle_body,
            coupler.needle_com,
            scratch,
            example.sim_dt,
            implicit=True,
            trial=True,
        )
        if alm is not None:
            assert alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance)
        example.interaction.accept(example.state_0, coupler.needle_body)
        if alm is not None:
            alm.commit()
    finally:
        if alm is not None and alm._open:
            alm.discard()
        wp.copy(example.state_0.body_f, body_force)
    assert example.interaction._previous_tip is not None
    assert example.interaction.track_count == 0
    assert not example.interaction.punctured


@contextmanager
def contact(*, capture=False, capacity=4096, iterations=160):
    with _contact(iterations=iterations, absolute_tolerance=2e-9, relative_tolerance=1e-6) as (
        ex,
        c,
        native,
        _external,
    ):
        c.capture_fem = capture
        c.capture_coupling = capture
        pc = enable(c, capacity)
        seed_contact(ex, c)
        yield ex, c, native, pc


def prepare_forces(ex, c, pc):
    pc.begin_step()
    assert pc.begin_trials(ex.state_0, ex.control)
    c._candidate.assign(ex.state_0)
    c._candidate.body_f.zero_()
    c._forces.zero_()
    c._trial_load.zero_()
    c._previous_residual.zero_()
    graph = CapturedHostTrialForces(c)
    graph.prepare(ex.sim_dt)
    return graph


def transaction_snapshot(ex, c, native):
    x = ex.interaction
    return {
        "input": _snapshot(ex.state_0),
        "output": _snapshot(ex.state_1),
        "native": _native_state(native),
        "increment": ex.solver.velocity_increment.numpy().copy(),
        "load": c._load.numpy().copy(),
        "previous_tip": x._previous_tip.copy(),
        "previous_q": x._previous_q.copy(),
        "events": x.events,
        "status": x.status,
    }


@contextmanager
def articulated_contact():
    from test_grasp_scene import _scene

    with _scene() as (scene, _config):
        pose = scene.plan.pose(0).copy()
        pose[2] += 0.6 * scene.geometry.thickness
        scene._initialize_pose(pose)
        scene.interaction.puncture_force = 1.0
        scene.solver.linear_solver_tolerance = 1e-6
        scene.solver.refresh_tangent(scene.state_0.particle_q, scene.sim_dt)
        c = NeedleTissueCoupler(
            scene.model,
            scene.solver,
            scene.rigid_solver,
            scene.interaction,
            scene.needle_body,
            scene.model.body_com.numpy()[scene.needle_body],
            absolute_tolerance=2e-9,
            relative_tolerance=1e-6,
            max_iterations=160,
            relaxation=1.0,
            compact_rigid=False,
            capture_fem=True,
            capture_coupling=True,
            surface_preconditioner=True,
        )
        yield scene, c


class SurfaceCapturePreparationTests(unittest.TestCase):
    def test_articulated_initial_seed_closes_alm_without_advancing_physics(self):
        with articulated_contact() as (scene, c):
            before = _snapshot(scene.state_0)
            native = _native_state(scene.rigid_solver)
            alm = scene.interaction.tunnel_alm
            self.assertIsNotNone(alm)
            committed = (alm.committed_normal.numpy(), alm.committed_tangent.numpy())
            seed_contact(scene, c)
            self.assertFalse(alm._open)
            self.assertEqual(alm.dual_updates, 0)
            self.assertEqual(scene.interaction.track_count, 0)
            self.assertEqual(scene.interaction.events, ())
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(scene.state_0, name).numpy(), value)
            np.testing.assert_array_equal(_native_state(scene.rigid_solver), native)
            np.testing.assert_array_equal(alm.committed_normal.numpy(), committed[0])
            np.testing.assert_array_equal(alm.committed_tangent.numpy(), committed[1])

    def test_real_initial_contact_has_nonempty_valid_recorder_and_certificate(self):
        with contact() as (ex, c, _native, pc):
            pc.begin_step()
            self.assertTrue(pc.begin_trials(ex.state_0, ex.control))
            c._candidate.assign(ex.state_0)
            c._candidate.body_f.zero_()
            permissions = _NeedleTrialForces(ex.interaction)
            permissions.add_forces(c._candidate, c.needle_body, c.needle_com, c._forces, ex.sim_dt)
            pc.validate()
            self.assertEqual(int(permissions.fallback.numpy()[0]), 0)
            self.assertGreater(int(pc.anchors.count.numpy()[0]), 0)
            self.assertEqual(int(pc.valid.numpy()[0]), 1)
            self.assertEqual(ex.interaction.events, ())
            self.assertEqual(ex.interaction.track_count, 0)

    def test_recorder_rebinding_invalidates_force_signature_but_values_do_not(self):
        with contact() as (ex, c, _native, pc):
            graph = prepare_forces(ex, c, pc)
            original = graph.signature
            pc.anchors.matrices.zero_()
            graph.prepare(ex.sim_dt)
            self.assertEqual(graph.signature, original)
            old = pc.anchors.matrices
            pc.anchors.matrices = wp.clone(old)
            graph.prepare(ex.sim_dt)
            self.assertNotEqual(graph.signature, original)
            self.assertIs(graph.permissions._surface_recorder_owners[8], pc.anchors.matrices)


class SurfaceCaptureCudaTests(unittest.TestCase):
    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Requires conditional CUDA capture")

    def assert_graphs_ran(self, c, pc, *, pc_graph=True):
        graph = c._host_trial_forces
        self.assertIsNotNone(graph)
        self.assertGreater(graph.launches, 0)
        self.assertGreater(graph.captures, 0)
        self.assertEqual(graph.fallbacks, 0)
        self.assertEqual(graph.result.size, 10)
        self.assertGreater(pc.applications, 0)
        if pc_graph:
            self.assertGreater(pc.graph_captures, 0)
            self.assertGreater(pc.graph_launches, 0)

    def assert_rollback(self, ex, c, native, before):
        for state, key in ((ex.state_0, "input"), (ex.state_1, "output")):
            for name, value in before[key].items():
                np.testing.assert_array_equal(getattr(state, name).numpy(), value, err_msg=name)
        np.testing.assert_array_equal(_native_state(native), before["native"])
        np.testing.assert_array_equal(ex.solver.velocity_increment.numpy(), before["increment"])
        np.testing.assert_array_equal(c._load.numpy(), before["load"])
        np.testing.assert_array_equal(ex.interaction._previous_tip, before["previous_tip"])
        np.testing.assert_array_equal(ex.interaction._previous_q, before["previous_q"])
        self.assertEqual(ex.interaction.events, before["events"])
        self.assertEqual(ex.interaction.status, before["status"])
        self.assertIsNone(ex.interaction._trial_fields)
        self.assertIsNone(ex.interaction._pending_opening)
        self.assertIsNone(ex.interaction._surface_tangent_anchors)
        self.assertFalse(c._surface_preconditioner.available)
        self.assertEqual(ex.solver.linear_solver_tolerance, 1e-6)

    def test_complete_captured_and_eager_contact_steps_match(self):
        endpoints = []
        for capture in (False, True):
            with (
                self.subTest(capture=capture),
                contact(capture=capture) as (ex, c, native, pc),
            ):
                c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assertEqual(ex.interaction.track_count, 0)
                self.assertEqual(ex.interaction.events, ())
                self.assertEqual(ex.interaction.status, "contact")
                self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)
                self.assertLessEqual(
                    c.residual,
                    c.absolute_tolerance
                    + c.relative_tolerance
                    * max(
                        np.max(np.abs(c._value.numpy())),
                        np.max(np.abs(c._trial_load.numpy())),
                    ),
                )
                if capture:
                    self.assert_graphs_ran(c, pc)
                    self.assertGreater(c.fem_graph_launches, 0)
                endpoints.append(_snapshot(ex.state_1))
        for name in (
            "particle_q",
            "particle_qd",
            "body_q",
            "body_qd",
            "joint_q",
            "joint_qd",
        ):
            # Same strict physical tolerance and established contact-fixture bound.
            np.testing.assert_allclose(
                endpoints[0][name],
                endpoints[1][name],
                rtol=2e-5,
                atol=3e-7,
                err_msg=name,
            )

    def test_captured_packet_overflow_falls_back_only_numerically(self):
        results = []
        for capacity in (4096, 0):
            with (
                self.subTest(capacity=capacity),
                contact(capture=True, capacity=capacity) as (ex, c, _native, pc),
            ):
                graph = prepare_forces(ex, c, pc)
                packet = graph.evaluate(ex.sim_dt, False)
                self.assertEqual(packet.shape, (8,))
                np.testing.assert_array_equal(packet[:6], c._statistics.numpy())
                self.assertGreater(packet[6], 0)
                self.assertEqual(packet[7], int(capacity > 0))
                self.assertEqual(graph.fallbacks, 0)
                self.assertEqual(graph.launches, 1)
                results.append(
                    (
                        c._forces.numpy(),
                        c._candidate.body_f.numpy(),
                        ex.interaction._contact_flags.numpy(),
                    )
                )
        for actual, expected in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_force_graph_recorder_rebinding_uses_new_owners(self):
        with contact(capture=True) as (ex, c, _native, pc):
            graph = prepare_forces(ex, c, pc)
            packet = graph.evaluate(ex.sim_dt, False)
            expected_forces = c._forces.numpy().copy()
            old_signature, captures = graph.signature, graph.captures
            old_count, old_matrices = pc.anchors.count, pc.anchors.matrices
            pc.anchors.count = wp.clone(old_count)
            pc.anchors.matrices = wp.clone(old_matrices)
            graph.prepare(ex.sim_dt)
            self.assertNotEqual(graph.signature, old_signature)
            self.assertFalse(graph.graphs)
            old_count.fill_(999)
            old_matrices.fill_(float("nan"))
            rebound = graph.evaluate(ex.sim_dt, False)
            self.assertEqual(graph.captures, captures + 1)
            np.testing.assert_array_equal(rebound, packet)
            np.testing.assert_array_equal(c._forces.numpy(), expected_forces)
            np.testing.assert_array_equal(old_count.numpy(), [999])
            self.assertTrue(np.isnan(old_matrices.numpy()).all())
            self.assertEqual(int(pc.valid.numpy()[0]), 1)

    def test_late_certificate_fallback_restarts_with_identical_physical_acceptance(
        self,
    ):
        results = []
        for capture in (False, True):
            with (
                self.subTest(capture=capture),
                contact(capture=capture) as (ex, c, native, pc),
            ):
                if capture:
                    with patch.object(ex.interaction, "_stage_certified_trial", return_value=False):
                        c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                    self.assertGreater(c._host_trial_forces.launches, 0)
                    self.assertGreater(c._host_trial_forces.fallbacks, 0)
                    self.assertGreater(pc.graph_launches, 0)
                else:
                    c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + ex.sim_dt, places=12)
                self.assertIsNone(ex.interaction._trial_fields)
                self.assertIsNone(ex.interaction._pending_opening)
                self.assertEqual(ex.interaction.events, ())
                self.assertEqual(ex.interaction.track_count, 0)
                self.assertEqual(ex.interaction.status, "contact")
                results.append(_snapshot(ex.state_1))
        for name in (
            "particle_q",
            "particle_qd",
            "body_q",
            "body_qd",
            "joint_q",
            "joint_qd",
        ):
            np.testing.assert_allclose(results[0][name], results[1][name], rtol=2e-5, atol=3e-7, err_msg=name)

    def test_nonfinite_candidate_rejects_before_geometry_or_stale_packet_use(self):
        with contact(capture=True) as (ex, c, _native, pc):
            graph = prepare_forces(ex, c, pc)
            graph.evaluate(ex.sim_dt, False)
            previous = (
                ex.interaction._device_queries.low.numpy(),
                ex.interaction._device_queries.high.numpy(),
            )
            q = c._candidate.particle_q.numpy()
            q[0, 0] = np.nan
            c._candidate.particle_q.assign(q)
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                graph.evaluate(ex.sim_dt, False)
            np.testing.assert_array_equal(ex.interaction._device_queries.low.numpy(), previous[0])
            np.testing.assert_array_equal(ex.interaction._device_queries.high.numpy(), previous[1])
            self.assertEqual(graph.launches, 2)

    def test_captured_budget_and_late_rejection_restore_every_transaction(self):
        for failure in ("budget", "late_tet"):
            with (
                self.subTest(failure=failure),
                contact(capture=True, iterations=1 if failure == "budget" else 160) as (
                    ex,
                    c,
                    native,
                    pc,
                ),
            ):
                before = transaction_snapshot(ex, c, native)
                if failure == "budget":
                    with self.assertRaises(NeedleCouplingNotConverged):
                        c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                else:
                    with (
                        patch.object(
                            c._tet_step_guard,
                            "check",
                            side_effect=NeedleStepTooLarge("late test rejection"),
                        ),
                        self.assertRaises(NeedleStepTooLarge),
                    ):
                        c.step(ex.state_0, ex.state_1, ex.control, ex.sim_dt)
                self.assert_graphs_ran(c, pc, pc_graph=failure != "budget")
                self.assert_rollback(ex, c, native, before)

    def test_pc_graph_rebinding_preserves_old_output_and_reads_new_residual(self):
        with contact(capture=True) as (ex, c, _native, pc):
            graph = prepare_forces(ex, c, pc)
            graph.evaluate(ex.sim_dt, False)
            pc.active = True
            for _ in range(3):
                self.assertTrue(pc.apply(c._candidate, ex.sim_dt, c._residual, c._direction))
            self.assertGreater(pc.graph_launches, 0)
            expected = c._direction.numpy().copy()
            old_key, old_captures = pc._graph_key, pc.graph_captures
            old_output = c._direction
            old_output.fill_(123.0)
            residual = wp.clone(c._residual)
            residual.assign(2.0 * c._residual.numpy())
            output = wp.empty_like(c._direction)
            for _ in range(2):
                self.assertTrue(pc.apply(c._candidate, ex.sim_dt, residual, output))
            self.assertNotEqual(pc._graph_key, old_key)
            self.assertGreater(pc.graph_captures, old_captures)
            np.testing.assert_array_equal(old_output.numpy(), np.full(old_output.size, 123.0))
            np.testing.assert_array_equal(output.numpy(), 2.0 * expected)

    def test_articulated_graph_uses_one_metric_per_transaction(self):
        with articulated_contact() as (scene, c):
            seed_contact(scene, c)
            pc = c._surface_preconditioner
            with patch.object(
                surface_preconditioner,
                "_native_needle_mass",
                wraps=surface_preconditioner._native_needle_mass,
            ) as metric:
                c.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
                self.assertEqual(metric.call_count, 1)
            self.assert_graphs_ran(c, pc)
            self.assertGreater(pc.applications, 1)
            self.assertEqual(scene.rigid_solver.rigid_substeps, 4)
            self.assertEqual(scene.rigid_solver.mj_model.neq, 0)
            self.assertEqual(scene.interaction.events, ())
            self.assertEqual(scene.interaction.track_count, 0)


if __name__ == "__main__":
    unittest.main()
