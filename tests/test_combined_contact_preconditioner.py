"""Independent combined-contact algebra, fallback and transactional contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import dense_action, history, setup
from test_needle_tissue_coupler import _native_state
from test_surface_anderson_coupling import _contact, _snapshot

from newton_solver_suturebot._src.solvers.coupled.acceleration.combined_contact_preconditioner import (
    CombinedContactPreconditioner,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import _ShaftPreconditioner
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_contact_tangent import SurfaceTangentAnchors
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_preconditioner import (
    SurfaceContactPreconditioner,
)
from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleCouplingNotConverged, NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco import SubcycledMuJoCo


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def jacobians(fixed, nodes, bary, levers):
    full, free = [], []
    n = len(fixed)
    for ids, weights, lever in zip(nodes, bary, levers, strict=True):
        j = np.zeros((3, 3 * (n + 2)))
        j[:, -6:-3] = np.eye(3)
        j[:, -3:] = -skew(lever)
        for node, weight in zip(ids, weights, strict=True):
            if node >= 0:
                j[:, 3 * node : 3 * node + 3] -= weight * np.eye(3)
        full.append(j)
        j = j.copy()
        j[:, np.flatnonzero(np.repeat(fixed, 3))] = 0
        free.append(j)
    return np.asarray(full), np.asarray(free)


def surface_fixture(ex):
    a = SurfaceTangentAnchors(4, ex.device)
    fixed = np.asarray(ex.interaction.fixed, bool)
    fixed_node, free_node = np.flatnonzero(fixed)[0], np.flatnonzero(~fixed)[0]
    nodes = np.array([[free_node, fixed_node, 3], [free_node, -1, 5]], np.int32)
    bary = np.array([[0.2, 0.3, 0.5], [0.25, 0.25, 0.5]], np.float32)
    levers = np.array([[0.2, -0.4, 0.3], [-0.2, 0.1, 0.6]], np.float32)
    matrices = np.array(
        [
            [[0.08, 0.01, 0.0], [0.01, 0.1, 0.02], [0.0, 0.02, 0.06]],
            [[0.1, -0.02, 0.01], [-0.02, 0.06, 0.0], [0.01, 0.0, 0.04]],
        ],
        np.float32,
    )
    a.count.assign(np.array([2], np.int32))
    for name, value in (("nodes", nodes), ("bary", bary), ("matrices", matrices), ("levers", levers)):
        full = np.zeros_like(getattr(a, name).numpy())
        full[:2] = value
        getattr(a, name).assign(full)
    return SimpleNamespace(anchors=a, count=4, valid=wp.ones(1, dtype=int, device=ex.device))


def native_embedded(ex, *, enabled=True, iterations=160):
    h = 1 / 2400
    setup(ex, h, 6000)
    native = SubcycledMuJoCo(
        ex.model,
        use_mujoco_cpu=True,
        rigid_substeps=4,
        iterations=200,
        ls_iterations=50,
        tolerance=1e-12,
        integrator="implicitfast",
        cone="elliptic",
    )
    native.reset(ex.state_0, flags=0)
    native.mj_data.time = 3.25
    ex.solver.linear_solver_tolerance = 1e-6
    c = NeedleTissueCoupler(
        ex.model,
        ex.solver,
        native,
        ex.interaction,
        ex.needle_body,
        ex.drive.com,
        moment_length=ex.geometry.radius,
        shaft_preconditioner=True,
        surface_preconditioner=enabled,
        preconditioned_anderson=True,
        compact_rigid=False,
        max_iterations=iterations,
        absolute_tolerance=2e-9,
        relative_tolerance=1e-6,
    )
    return h, native, c


class CombinedContactAlgebraTests(unittest.TestCase):
    def fixture(self, ex):
        h = 1 / 240
        setup(ex, h, 6000)
        surface = surface_fixture(ex)
        pc = CombinedContactPreconditioner(ex.coupler, surface)
        self.assertTrue(pc.active)
        return h, surface, pc

    def test_combined_action_diagonal_and_arbitrary_residual_match_dense_sum(self):
        with _example() as ex:
            h, surface, pc = self.fixture(ex)
            rng = np.random.default_rng(907)
            r = rng.normal(0, 1e-5, 3 * pc.n)
            residual = wp.array(r, dtype=wp.float64, device=ex.device)
            out = wp.empty_like(residual)
            before = history(ex.interaction)
            for scale in (0.0, 1.0, -3.0):
                residual.assign(scale * r)
                pc.apply(ex.state_0, h, residual, out)
                mass = dense_action(pc, "mass")
                # The track arrays and the anchors have the rows of the track reserve; the live ones hold the samples.
                count = ex.interaction.track_count
                shaft_full, shaft_free = jacobians(
                    ex.interaction.fixed,
                    ex.interaction._track_nodes.numpy()[:count],
                    ex.interaction._track_bary.numpy()[:count],
                    pc.levers.numpy()[:count],
                )
                a = surface.anchors
                surface_full, surface_free = jacobians(
                    ex.interaction.fixed, a.nodes.numpy()[:2], a.bary.numpy()[:2], a.levers.numpy()[:2]
                )
                full = np.concatenate((shaft_full, surface_full))
                free = np.concatenate((shaft_free, surface_free))
                matrices = np.concatenate((pc.matrices.numpy()[:count], a.matrices.numpy()[:2])).astype(float)
                h32 = float(np.float32(h))
                stiffness = sum(h32 * j.T @ d @ j for j, d in zip(free, matrices, strict=True))
                balanced = sum(h32 * ja.T @ d @ jf for ja, jf, d in zip(full, free, matrices, strict=True))
                operator = dense_action(pc, "operator")
                np.testing.assert_allclose(operator, mass + stiffness, rtol=4e-6, atol=2e-9)
                expected_diag = np.asarray(
                    [(mass + stiffness)[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] for i in range(pc.n)]
                )
                np.testing.assert_allclose(pc.diagonal.numpy(), expected_diag, rtol=4e-6, atol=2e-9)
                # Compare correction using the actual inexact CG output; fixed
                # and unmapped support reactions must not be dropped.
                expected = scale * r - balanced @ pc.z.numpy().reshape(-1)
                np.testing.assert_allclose(out.numpy(), expected, rtol=4e-5, atol=2e-10)
                # Independent exact dense solve gives a certified discrepancy:
                # delta_z=A^-1 linear_residual, not an arbitrary state tolerance.
                b = (scale * r).astype(np.float32).astype(float)
                exact_z = np.linalg.solve(mass + stiffness, b)
                inner_residual = b - (mass + stiffness) @ pc.z.numpy().reshape(-1)
                predicted_error = balanced @ np.linalg.solve(mass + stiffness, inner_residual)
                np.testing.assert_allclose(
                    expected - (scale * r - balanced @ exact_z), predicted_error, rtol=1e-7, atol=2e-16
                )
            self.assertEqual(history(ex.interaction), before)
            if ex.device.is_cuda:
                self.assertGreater(pc.graph_launches, 0)

    def test_invalid_surface_recording_is_exact_shaft_only_and_storage_invalidates_capture(self):
        with _example() as ex:
            h, surface, pc = self.fixture(ex)
            original = _ShaftPreconditioner(ex.coupler)
            rng = np.random.default_rng(57)
            r = wp.array(rng.normal(0, 1e-5, 3 * pc.n), dtype=wp.float64, device=ex.device)
            out, reference = wp.empty_like(r), wp.empty_like(r)
            surface.valid.zero_()
            for _ in range(3):
                pc.apply(ex.state_0, h, r, out)
                original.apply(ex.state_0, h, r, reference)
                np.testing.assert_array_equal(out.numpy(), reference.numpy())
            old_key = pc._graph_key
            old_owners = pc._graph_inputs
            surface.anchors.matrices = wp.clone(surface.anchors.matrices)
            pc.apply(ex.state_0, h, r, out)
            np.testing.assert_array_equal(out.numpy(), reference.numpy())
            if ex.device.is_cuda:
                self.assertNotEqual(pc._graph_key, old_key)
                self.assertIn(surface.anchors.matrices, pc._graph_inputs)
                self.assertTrue(old_owners)
            # Device validity changes are consumed by replay, without rebuilding
            # on every contact count change or silently reusing a stale metric.
            surface.valid.fill_(1)
            pc.apply(ex.state_0, h, r, out)
            self.assertGreater(np.linalg.norm(out.numpy() - reference.numpy()), 1e-8)

    def test_timestep_changes_replay_the_preparation_and_solve_graphs_without_recapture(self):
        # The contact metric h Jᵀ D J, the anchor tangents and the rigid damping term use the
        # time step h, which the kernels read from the coupler's step time. The graph of the
        # input preparation and the graph of the solve are therefore captured once and replay
        # at a halved step and at a frame remainder. The reference issues the same application
        # eagerly: its solve is given a new key before every call and its preparation is not
        # captured.
        with _example() as ex:
            h, surface, pc = self.fixture(ex)
            reference = CombinedContactPreconditioner(ex.coupler, surface)
            reference.capture_preparation = False
            rng = np.random.default_rng(311)
            r = wp.array(rng.normal(0, 1e-5, 3 * pc.n), dtype=wp.float64, device=ex.device)
            out, expected = wp.empty_like(r), wp.empty_like(r)
            corrections = {}
            for index, dt in enumerate((h, h, h, h / 2, 0.37 * h, h, h / 2)):
                pc.apply(ex.state_0, dt, r, out)
                actual = out.numpy().copy()
                reference._graph_key = None
                reference.apply(ex.state_0, dt, r, expected)
                np.testing.assert_array_equal(actual, expected.numpy(), err_msg=f"application {index}")
                np.testing.assert_array_equal(pc.time_step.numpy(), [np.float32(dt)])
                self.assertIs(pc.step_time, ex.coupler.step_time)
                if ex.device.is_cuda:
                    # The first application is issued as written, the second is captured.
                    replay = pc._input_preparation.replay
                    self.assertEqual((pc.graph_captures, pc.graph_launches), (min(index, 1), index))
                    self.assertEqual((replay.eager, replay.captures, replay.launches), (1, min(index, 1), index))
                    self.assertEqual((reference.graph_captures, reference.graph_launches), (0, 0))
                    self.assertEqual(reference._input_preparation.replay.launches, 0)
                corrections.setdefault(dt, actual)
            # The correction depends on the time step, so a graph that kept the time step of
            # its capture would differ from the eager application above.
            for dt in (h / 2, 0.37 * h):
                self.assertGreater(np.abs(corrections[dt] - corrections[h]).max(), 0.01 * np.abs(corrections[h]).max())

    def test_postpuncture_recorder_eligibility_preserves_gradient_guards(self):
        with _contact() as (ex, c, native, external):
            pc = SurfaceContactPreconditioner(c, 4)
            ex.interaction.punctured = True
            self.assertFalse(pc.begin_trials(ex.state_0, ex.control))
            self.assertTrue(pc.begin_trials(ex.state_0, ex.control, include_history=True))
            ex.state_0.particle_qd.requires_grad = True
            self.assertFalse(pc.begin_trials(ex.state_0, ex.control, include_history=True))
            self.assertIsNone(ex.interaction._surface_tangent_anchors)
            self.assertEqual(pc.valid.numpy()[0], 0)

    def test_recording_preserves_actual_postpuncture_surface_and_shaft_forces(self):
        with _example() as ex:
            h, native, c = native_embedded(ex)
            pc = c._surface_preconditioner
            c._candidate.assign(ex.state_0)
            before = history(ex.interaction)
            c._evaluate(c._candidate, h, trial=True)
            original = (
                c._forces.numpy().copy(),
                c._candidate.body_f.numpy().copy(),
                ex.interaction._diagnostics.numpy().copy(),
            )
            self.assertFalse(pc.begin_trials(ex.state_0, ex.control))
            self.assertTrue(pc.begin_trials(ex.state_0, ex.control, include_history=True))
            c._evaluate(c._candidate, h, trial=True)
            pc.validate()
            self.assertEqual(pc.valid.numpy()[0], 1)
            self.assertGreater(pc.anchors.count.numpy()[0], 0)
            self.assertGreater(ex.interaction.engaged_track_length, 0)
            for got, expected in zip(
                (c._forces.numpy(), c._candidate.body_f.numpy(), ex.interaction._diagnostics.numpy()),
                original,
                strict=True,
            ):
                np.testing.assert_array_equal(got, expected)
            self.assertEqual(history(ex.interaction), before)

    def test_flag_of_the_combined_correction_waits_for_its_first_reader_and_the_surface_metric_never_reads(self):
        with _example() as ex:
            h, native, c = native_embedded(ex)
            combined, surface, interaction = c._shaft_preconditioner, c._surface_preconditioner, ex.interaction
            self.assertIs(type(combined), CombinedContactPreconditioner)
            self.assertIs(type(surface), SurfaceContactPreconditioner)
            # Neither construction read the engagement or asked for it.
            self.assertFalse(interaction._engaged_length_requested)
            with patch.object(
                interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length
            ) as read:
                self.assertIsNone(combined._active)
                for pc in (surface, combined):
                    pc.begin_step()
                self.assertEqual((combined._active, read.call_count), (None, 0))
                # The surface metric holds the flag that its owner assigns; it has no engagement to read.
                self.assertIs(surface.active, False)
                surface.active = True
                self.assertIs(surface.active, True)
                surface.begin_step()
                self.assertIs(surface.active, False)
                self.assertEqual(read.call_count, 0)
                # The combined correction evaluates at its first reader, as the shaft correction does.
                self.assertIs(combined.active, True)
                self.assertIs(combined.active, True)
                self.assertEqual(read.call_count, 1)
                combined.active = False
                self.assertIs(combined.active, False)
                combined.begin_step()
                self.assertIsNone(combined._active)
                # A step of the host loop: its trials are the first reader, once, and the acceptance
                # reads the length for the step after it.
                interaction._engaged_length_requested = False
                solve, seen = c._solve_host, []

                def host(*args, **kwargs):
                    seen.append((args[5], combined._active, read.call_count))
                    return solve(*args, **kwargs)

                with patch.object(c, "_solve_host", side_effect=host):
                    c.step(ex.state_0, ex.state_1, ex.control, h)
                self.assertEqual(seen, [(True, True, 2)])
                self.assertEqual(read.call_count, 3)

    def test_real_combined_native_solve_keeps_same_physical_acceptance(self):
        records = []
        for enabled in (False, True):
            with self.subTest(enabled=enabled), _example() as ex:
                h, native, c = native_embedded(ex, enabled=enabled)
                c.step(ex.state_0, ex.state_1, ex.control, h)
                self.assertAlmostEqual(native.mj_data.time, 3.25 + h, places=12)
                self.assertLessEqual(
                    c.residual,
                    c.absolute_tolerance
                    + c.relative_tolerance
                    * max(np.max(np.abs(c._value.numpy())), np.max(np.abs(c._trial_load.numpy()))),
                )
                self.assertEqual(c.max_iterations, 160)
                self.assertEqual(ex.solver.linear_solver_tolerance, 1e-6)
                self.assertLess(ex.interaction.force_balance_error, 2e-8)
                self.assertLess(ex.interaction.torque_balance_error, 1e-9)
                if enabled:
                    self.assertGreater(c._surface_preconditioner.selection_changes, 0)
                records.append((_snapshot(ex.state_1), history(ex.interaction), ex.interaction.interface_force.copy()))
        self.assertEqual(records[0][1], records[1][1])
        for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_allclose(records[0][0][name], records[1][0][name], rtol=3e-5, atol=3e-7, err_msg=name)
        np.testing.assert_allclose(records[0][2], records[1][2], rtol=3e-5, atol=1e-8)

    def test_combined_rejection_restores_native_fem_and_history(self):
        with _example() as ex:
            h, native, c = native_embedded(ex, iterations=1)
            before_in, before_out = _snapshot(ex.state_0), _snapshot(ex.state_1)
            before_native, before_history = _native_state(native), history(ex.interaction)
            before_guess, before_load = ex.solver.velocity_increment.numpy().copy(), c._load.numpy().copy()
            with self.assertRaises(NeedleCouplingNotConverged):
                c.step(ex.state_0, ex.state_1, ex.control, h)
            self.assertGreater(c._surface_preconditioner.selection_changes, 0)
            for state, before in ((ex.state_0, before_in), (ex.state_1, before_out)):
                for name, expected in before.items():
                    np.testing.assert_array_equal(getattr(state, name).numpy(), expected)
            np.testing.assert_array_equal(_native_state(native), before_native)
            np.testing.assert_array_equal(ex.solver.velocity_increment.numpy(), before_guess)
            np.testing.assert_array_equal(c._load.numpy(), before_load)
            self.assertEqual(history(ex.interaction), before_history)
            self.assertIsNone(ex.interaction._surface_tangent_anchors)


if __name__ == "__main__":
    unittest.main()
