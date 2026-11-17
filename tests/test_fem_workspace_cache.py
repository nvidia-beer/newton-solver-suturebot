# SPDX-License-Identifier: Apache-2.0
"""Adaptive inner tolerances reuse CUDA scratch/graphs without changing FEM."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import test_needle_coupling_graph as coupling_fixture
import warp as wp
from test_needle_insertion_example import _example, _prepare

from newton_solver_suturebot._src.utils.capture import ScopedCapture


def _registered(solver, workspace):
    return any(workspace is value for value in solver._linear_workspaces.values())


@contextmanager
def _cacheable(solver):
    """Give every workspace a registry key on the CPU, where the solver caches none.

    The registry's bookkeeping (pins, eviction, drop, epoch) is host logic. On a CUDA
    device the solver's own keys are used.
    """
    if solver.model.device.is_cuda:
        yield
        return

    def key(operator, inverse, settings):
        return ("host test", settings, id(operator), id(inverse))

    with patch.object(solver, "_linear_workspace_key", side_effect=key):
        yield


class WorkspaceRegistryTests(unittest.TestCase):
    def test_tolerance_levels_are_the_floats_that_host_controlled_trials_reach(self):
        with _example() as example:
            solver = example.solver
            # At 7e-3 and 6e-4 a division by ten gives other floats than the product with 0.1.
            for configured in (3e-3, 1e-2, 7e-3, 6e-4, 1e-4, 1e-6, 2.5e-7):
                solver.linear_solver_tolerance = configured
                # NeedleTissueCoupler._solve_host tightens the tolerance by this expression.
                expected = [configured]
                while True:
                    tighter = max(min(configured, 1e-6), expected[-1] * 0.1)
                    if not tighter < expected[-1]:
                        break
                    expected.append(tighter)
                self.assertEqual(solver.tolerance_levels(), expected)
                self.assertEqual(solver.tolerance_levels(configured), expected)
                self.assertEqual(solver.linear_solver_tolerance, configured)
            solver.linear_solver_tolerance = 3e-3
            self.assertEqual(len(solver.tolerance_levels()), 5)
            self.assertEqual(solver.tolerance_levels()[-1], 1e-6)
            self.assertEqual(solver.tolerance_levels(1e-6), [1e-6])

    def test_pinned_level_workspaces_survive_ad_hoc_tolerances_and_the_rest_stays_bounded(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, state, dt = example.solver, example.state_0, example.sim_dt
            output = example.model.state()
            solver.refresh_tangent(state.particle_q, dt)

            def step(tolerance):
                solver.linear_solver_tolerance = tolerance
                solver.step(state, output, example.control, None, dt)

            with _cacheable(solver):
                configured = solver.linear_solver_tolerance
                step(configured)
                first = solver._lin_solver
                self.assertTrue(_registered(solver, first))
                epoch = solver.linear_binding_epoch()
                levels = solver.tolerance_levels()
                self.assertEqual(len(levels), 5)
                with patch.object(solver, "_make_linear_solver", wraps=solver._make_linear_solver) as allocate:
                    pinned = [solver.linear_workspace(value) for value in levels]
                    # The level of the step above is the workspace that step made.
                    self.assertEqual(allocate.call_count, len(levels) - 1)
                    self.assertIs(pinned[0][1], first)
                    again = [solver.linear_workspace(value) for value in levels]
                    self.assertEqual(allocate.call_count, len(levels) - 1)
                for (settings, workspace), (same_settings, same) in zip(pinned, again, strict=True):
                    self.assertEqual(settings, same_settings)
                    self.assertIs(workspace, same)
                self.assertEqual([settings[3] for settings, _ in pinned], levels)
                self.assertEqual(len(solver._pinned_workspaces), len(levels))
                # Fetching neither selects a workspace nor moves the epoch.
                self.assertIs(solver._lin_solver, first)
                self.assertEqual(solver.linear_binding_epoch(), epoch)
                # Forty tolerances that are no level: each step makes its own workspace.
                for index in range(40):
                    step(configured * (0.9 - 0.01 * index))
                    self.assertFalse(any(solver._lin_solver is workspace for _, workspace in pinned))
                    for _, workspace in pinned:
                        self.assertTrue(_registered(solver, workspace), index)
                    self.assertLessEqual(len(solver._linear_workspaces) - len(pinned), 8)
                self.assertEqual(len(solver._linear_workspaces), len(pinned) + 8)
                self.assertEqual(len(solver._pinned_workspaces), len(levels))
                # A step at a level selects the pinned workspace of that level, without a new one.
                with patch.object(solver, "_make_linear_solver", wraps=solver._make_linear_solver) as allocate:
                    for value, (settings, workspace) in zip(levels, pinned, strict=True):
                        step(value)
                        self.assertIs(solver._lin_solver, workspace)
                        self.assertEqual(solver._lin_solver_key, settings)
                    self.assertEqual(allocate.call_count, 0)
                self.assertEqual(solver.linear_binding_epoch(), epoch)
                # Released workspaces stay registered until other tolerances push them out.
                solver.unpin_linear_workspaces([workspace for _, workspace in pinned[1:]])
                self.assertEqual(len(solver._pinned_workspaces), 1)
                for _, workspace in pinned:
                    self.assertTrue(_registered(solver, workspace))
                for index in range(40):
                    step(configured * (0.5 - 0.005 * index))
                    self.assertLessEqual(len(solver._linear_workspaces) - 1, 8)
                self.assertTrue(_registered(solver, pinned[0][1]))
                for _, workspace in pinned[1:]:
                    self.assertFalse(_registered(solver, workspace))
                self.assertEqual(solver.linear_binding_epoch(), epoch)
                # An explicit invalidation drops every workspace, the pinned one included, and
                # tells the owner of a binding so; a fetch after it makes new workspaces.
                solver._lin_solver = None
                self.assertEqual(solver.linear_binding_epoch(), epoch + 1)
                self.assertFalse(solver._linear_workspaces)
                self.assertFalse(solver._pinned_workspaces)
                renewed = [solver.linear_workspace(value) for value in levels]
                for (_, old), (_, new) in zip(pinned, renewed, strict=True):
                    self.assertIsNot(new, old)
                # Nothing is selected yet, so the next question drops the registry once more.
                self.assertEqual(solver.linear_binding_epoch(), epoch + 2)
                self.assertFalse(solver._linear_workspaces)
                solver.linear_solver_tolerance = configured

    def test_released_level_workspaces_leave_without_an_eager_step_and_the_selected_one_stays(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, state, dt = example.solver, example.state_0, example.sim_dt
            output = example.model.state()
            solver.refresh_tangent(state.particle_q, dt)

            def rebind(bound, tolerance):
                # What the preparation of the trial graph does at a changed configured tolerance:
                # fetch the workspaces of its levels and release those of the levels before.
                fetched = [solver.linear_workspace(value) for value in solver.tolerance_levels(tolerance)]
                stale = [w for _, w in bound if all(w is not kept for _, kept in fetched)]
                if stale:
                    solver.unpin_linear_workspaces(stale)
                self.assertEqual(len(solver._pinned_workspaces), len(fetched))
                self.assertLessEqual(len(solver._linear_workspaces) - len(fetched), 8)
                for _, workspace in fetched:
                    self.assertTrue(_registered(solver, workspace))
                return fetched

            with _cacheable(solver):
                configured = solver.linear_solver_tolerance
                # An eager step at a tolerance that is no level: its workspace is selected and has no pin.
                solver.linear_solver_tolerance = configured * 0.37
                solver.step(state, output, example.control, None, dt)
                selected, epoch = solver._lin_solver, solver.linear_binding_epoch()
                bound = []
                # No eager step registers a workspace from here on: only the releases evict.
                for index in range(12):
                    with self.subTest(round=index, selection="eager"):
                        bound = rebind(bound, configured * (1.0 - 0.01 * index))
                        # The oldest workspace without a pin, and it stays: it is the selected one.
                        self.assertIs(solver._lin_solver, selected)
                        self.assertTrue(_registered(solver, selected))
                self.assertEqual(len(solver._linear_workspaces), len(bound) + 8)
                self.assertTrue(solver._reuse_linear_workspace())
                self.assertIs(solver._lin_solver, selected)
                self.assertEqual(solver.linear_binding_epoch(), epoch)
                # The trial graph selects the workspace of its first level after every binding.
                for index in range(12, 52):
                    with self.subTest(round=index, selection="first level"):
                        bound = rebind(bound, configured * (1.0 - 0.01 * index))
                        solver._lin_solver_key, solver._lin_solver = bound[0]
                self.assertEqual(len(solver._linear_workspaces), len(bound) + 8)
                self.assertFalse(_registered(solver, selected))
                solver.linear_solver_tolerance = configured * (1.0 - 0.01 * 51)
                self.assertTrue(solver._reuse_linear_workspace())
                self.assertIs(solver._lin_solver, bound[0][1])
                self.assertEqual(solver.linear_binding_epoch(), epoch)
                solver.linear_solver_tolerance = configured

    def test_explicit_invalidation_drops_the_registry_at_its_next_use(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, state, dt = example.solver, example.state_0, example.sim_dt
            output = example.model.state()
            solver.refresh_tangent(state.particle_q, dt)

            def bind():
                # What an owner of a graph does: ask for the epoch, fetch, and a step selects.
                solver.step(state, output, example.control, None, dt)
                epoch = solver.linear_binding_epoch()
                pinned = [solver.linear_workspace(value)[1] for value in solver.tolerance_levels()]
                self.assertIs(solver._lin_solver, pinned[0])
                self.assertEqual(len(solver._pinned_workspaces), len(pinned))
                return epoch, pinned

            def dropped(epoch, pinned):
                self.assertEqual(solver._workspace_epoch, epoch + 1)
                self.assertFalse(solver._pinned_workspaces)
                for workspace in pinned:
                    self.assertFalse(_registered(solver, workspace))

            with _cacheable(solver):
                # A registered selection is reused, at the tolerance that is configured now.
                epoch, pinned = bind()
                configured = solver.linear_solver_tolerance
                solver.linear_solver_tolerance = solver.tolerance_levels()[2]
                self.assertTrue(solver._reuse_linear_workspace())
                self.assertIs(solver._lin_solver, pinned[2])
                self.assertEqual(solver.linear_binding_epoch(), epoch)
                solver.linear_solver_tolerance = configured
                # The next step after an invalidation makes a new workspace and keeps no old one.
                solver._lin_solver = None
                solver.step(state, output, example.control, None, dt)
                dropped(epoch, pinned)
                self.assertEqual(len(solver._linear_workspaces), 1)
                self.assertTrue(_registered(solver, solver._lin_solver))
                # The selection of a workspace before a graph lookup honors it as well.
                epoch, pinned = bind()
                solver._lin_solver = None
                self.assertFalse(solver._reuse_linear_workspace())
                dropped(epoch, pinned)
                self.assertFalse(solver._linear_workspaces)
                # So does a selected workspace that is not the registry's.
                epoch, pinned = bind()
                operator = solver.A_bsr if solver._fixed_particles is None else solver._fixed_particles.operator
                solver._lin_solver = solver._make_linear_solver(
                    operator, solver.M_bsr, solver._linear_solver_settings()
                )
                self.assertFalse(solver._reuse_linear_workspace())
                dropped(epoch, pinned)
                self.assertFalse(solver._linear_workspaces)

    def test_configuration_without_a_registry_key_gets_a_new_workspace_per_call(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver = example.solver
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            # Host residual checks are one of the configurations the registry does not hold.
            solver.linear_solver_check_every = 1
            first, second = solver.linear_workspace(), solver.linear_workspace()
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[0], solver._linear_solver_settings())
            self.assertIsNot(first[1], second[1])
            self.assertFalse(solver._linear_workspaces)
            self.assertFalse(solver._pinned_workspaces)
            solver.unpin_linear_workspaces([first[1], second[1]])
            self.assertFalse(solver._pinned_workspaces)


class FEMWorkspaceCacheTests(unittest.TestCase):
    def setUp(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Workspace and graph reuse applies to CUDA FEM")

    def test_revisiting_tolerances_timesteps_and_states_reuses_exact_physics_graphs(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, coupler = example.solver, example.coupler
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            expected = example.model.state()
            rng = np.random.default_rng(891)
            shape = (example.model.particle_count, 3)
            warmstart = rng.normal(0, 1e-8, shape).astype(np.float32)
            cases = [(dt, tol, state) for dt in (1.0, 0.5) for tol in (3e-3, 3e-4, 3e-5) for state in (0, 1)]
            initial_counts = None
            with patch.object(solver, "_make_linear_solver", wraps=solver._make_linear_solver) as allocate:
                for repeat in range(3):
                    for index, (scale, tolerance, buffer) in enumerate(cases):
                        with self.subTest(repeat=repeat, case=index):
                            state = (example.state_0, example.state_1)[buffer]
                            positions = example.selection.vertices.copy()
                            positions[:, 0] += (repeat + 1) * 1e-7 * np.arange(len(positions))
                            state.particle_q.assign(positions)
                            state.particle_qd.assign(rng.normal(0, 1e-5, shape).astype(np.float32))
                            dt = scale * example.sim_dt
                            solver.linear_solver_tolerance = tolerance
                            solver.refresh_tangent(state.particle_q, dt)
                            coupler._forces.assign(rng.normal(0, 1e-5, shape).astype(np.float32))
                            solver.velocity_increment.assign(warmstart)
                            expected.assign(state)
                            solver.step(state, expected, example.control, None, dt, external_forces=coupler._forces)
                            increment = solver.velocity_increment.numpy().copy()
                            solver.velocity_increment.assign(warmstart)
                            coupler._candidate.assign(state)
                            coupler._step_fem(state, example.control, dt)
                            for name in ("particle_q", "particle_qd"):
                                np.testing.assert_allclose(
                                    getattr(coupler._candidate, name).numpy(),
                                    getattr(expected, name).numpy(),
                                    rtol=1e-6,
                                    atol=2e-9,
                                )
                            np.testing.assert_allclose(
                                solver.velocity_increment.numpy(), increment, rtol=1e-6, atol=2e-9
                            )
                            self.assertEqual(solver.linear_solver_tolerance, tolerance)
                    counts = (allocate.call_count, coupler.fem_graph_captures)
                    if initial_counts is None:
                        initial_counts = counts
                        # One workspace per tolerance, one graph per tolerance and state buffer.
                        # The time step is device data: the cases at the halved step replay the
                        # graphs that the full step captured.
                        self.assertEqual(counts, (3, len(cases) // 2))
                    else:
                        self.assertEqual(counts, initial_counts)
            self.assertEqual(coupler.fem_graph_launches, 3 * len(cases))

    def test_replaced_workspace_is_not_resurrected_and_cache_memory_is_bounded(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver, coupler = example.solver, example.coupler
            state, dt = example.state_0, example.sim_dt
            solver.refresh_tangent(state.particle_q, dt)
            for _ in range(2):
                coupler._step_fem(state, example.control, dt)
            previous = solver._lin_solver
            solver._lin_solver = None
            coupler._step_fem(state, example.control, dt)
            self.assertIsNot(solver._lin_solver, previous)
            self.assertFalse(any(workspace is previous for workspace in solver._linear_workspaces.values()))
            previous = solver._lin_solver
            solver._lin_solver = None
            solver.step(state, coupler._candidate, example.control, None, dt, external_forces=coupler._forces)
            self.assertIsNot(solver._lin_solver, previous)
            self.assertFalse(any(workspace is previous for workspace in solver._linear_workspaces.values()))
            # Distinct timesteps and numerical levels retain only bounded
            # scratch and graph history, even for a caller with no steady cycle.
            for index in range(40):
                solver.linear_solver_tolerance = 0.003 / (1 + index)
                trial_dt = dt * (1 + index / 100)
                solver.refresh_tangent(state.particle_q, trial_dt)
                coupler._step_fem(state, example.control, trial_dt)
                coupler._step_fem(state, example.control, trial_dt)
                self.assertLessEqual(len(solver._linear_workspaces), 8)
                self.assertLessEqual(len(coupler._fem_graphs), 32)
            coupler.invalidate_fem_graph()
            self.assertFalse(coupler._fem_graphs)

    def test_workspace_is_not_fetched_inside_a_capture(self):
        with _example() as example:
            _prepare(example)
            example.start()
            solver = example.solver
            solver.refresh_tangent(example.state_0.particle_q, example.sim_dt)
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                with ScopedCapture(device=example.model.device):
                    example.coupler._forces.zero_()
                    solver.linear_workspace()
            self.assertFalse(example.model.device.is_capturing)
            self.assertFalse(solver._pinned_workspaces)

    def test_host_trials_between_device_preparations_replay_graphs_of_the_registered_level_workspaces(self):
        # The trial graph binds one Krylov workspace per tolerance level, fetched from the
        # solver's registry. A host-controlled FEM trial at any of these levels selects the same
        # workspace, so after one capture per level and state buffer it replays a graph instead
        # of stepping eagerly, whatever the trial graph prepared in between.
        dt = coupling_fixture._DT
        with coupling_fixture._pair("penalty") as (_, example):
            coupler, solver = example.coupler, example.solver
            coupler.step(example.state_0, example.state_1, example.control, dt)
            trials = coupler._captured_trials
            self.assertEqual((coupler.coupling_graph_launches, coupler.coupling_graph_fallbacks), (1, 0))
            levels = solver.tolerance_levels()
            self.assertEqual(len(levels), 5)
            bound = [workspace for _, workspace in trials.linear_solvers]
            self.assertEqual([settings[3] for settings, _ in trials.linear_solvers], levels)
            for workspace in bound:
                self.assertTrue(_registered(solver, workspace))
            self.assertEqual(len(solver._pinned_workspaces), len(levels))
            self.assertTrue(_registered(solver, solver._lin_solver))
            epoch = solver.linear_binding_epoch()
            self.assertEqual(trials.signature[-1], epoch)

            expected = example.model.state()
            rng = np.random.default_rng(417)
            shape = (example.model.particle_count, 3)
            warmstart = rng.normal(0, 1e-8, shape).astype(np.float32)
            positions = example.state_0.particle_q.numpy().copy()
            captures, launches = coupler.fem_graph_captures, coupler.fem_graph_launches
            trials_run = 0
            with patch.object(solver, "_make_linear_solver", wraps=solver._make_linear_solver) as allocate:
                for repeat in range(3):
                    for buffer in (0, 1):
                        state = (example.state_0, example.state_1)[buffer]
                        for level, tolerance in enumerate(levels):
                            with self.subTest(repeat=repeat, buffer=buffer, level=level):
                                moved = positions.copy()
                                moved[:, 0] += (repeat + 1) * 1e-7 * np.arange(len(moved))
                                state.particle_q.assign(moved)
                                state.particle_qd.assign(rng.normal(0, 1e-5, shape).astype(np.float32))
                                # The preparation of the device trials, as at the start of an
                                # attempt: at the configured tolerance. Every other time its
                                # graphs are dropped first, so it warms, fetches and selects again.
                                solver.linear_solver_tolerance = levels[0]
                                dropped = bool((repeat + buffer + level) % 2)
                                if dropped:
                                    trials.invalidate()
                                trials.prepare(state, example.control, dt)
                                # Whatever the preparation did, a level workspace is selected; a
                                # preparation that had to bind again selects the first level.
                                self.assertTrue(any(solver._lin_solver is workspace for workspace in bound))
                                if dropped:
                                    self.assertIs(solver._lin_solver, bound[0])
                                self.assertEqual(trials.signature[-1], epoch)
                                for kept, (_, workspace) in zip(bound, trials.linear_solvers, strict=True):
                                    self.assertIs(workspace, kept)
                                solver.linear_solver_tolerance = tolerance
                                coupler._forces.assign(rng.normal(0, 1e-5, shape).astype(np.float32))
                                solver.velocity_increment.assign(warmstart)
                                expected.assign(state)
                                solver.step(state, expected, example.control, None, dt, external_forces=coupler._forces)
                                self.assertIs(solver._lin_solver, bound[level])
                                increment = solver.velocity_increment.numpy().copy()
                                solver.velocity_increment.assign(warmstart)
                                coupler._candidate.assign(state)
                                coupler._step_fem(state, example.control, dt)
                                trials_run += 1
                                self.assertEqual(coupler.fem_graph_status, "replaying")
                                self.assertIs(solver._lin_solver, bound[level])
                                self.assertTrue(_registered(solver, solver._lin_solver))
                                for name in ("particle_q", "particle_qd"):
                                    _assert_same_step(
                                        getattr(coupler._candidate, name).numpy(), getattr(expected, name).numpy(), name
                                    )
                                _assert_same_step(solver.velocity_increment.numpy(), increment, "increment")
                    if repeat == 0:
                        # One graph per level and state buffer, none after the first pass.
                        self.assertEqual(coupler.fem_graph_captures - captures, 2 * len(levels))
                self.assertEqual(allocate.call_count, 0)
            self.assertEqual(coupler.fem_graph_captures - captures, 2 * len(levels))
            self.assertEqual(coupler.fem_graph_launches - launches, trials_run)
            self.assertEqual(trials_run, 3 * 2 * len(levels))
            self.assertEqual(solver.linear_binding_epoch(), epoch)
            self.assertEqual(len(solver._pinned_workspaces), len(levels))
            # Another configured tolerance has other levels. The trial graph binds their
            # workspaces and releases those it no longer binds, which stay registered.
            solver.linear_solver_tolerance = 1e-3
            other = solver.tolerance_levels()
            self.assertFalse(set(other[:-1]) & set(levels))
            trials.prepare(example.state_0, example.control, dt)
            self.assertEqual([settings[3] for settings, _ in trials.linear_solvers], other)
            self.assertEqual(len(solver._pinned_workspaces), len(other))
            for workspace in bound:
                self.assertTrue(_registered(solver, workspace))
            self.assertIs(trials.linear_solvers[-1][1], bound[-1])
            self.assertIs(solver._lin_solver, trials.linear_solvers[0][1])
            self.assertEqual(trials.signature[-1], epoch)
            # Dropped graphs bind nothing: every workspace may leave the registry again.
            trials.invalidate()
            self.assertFalse(solver._pinned_workspaces)
            self.assertTrue(_registered(solver, solver._lin_solver))
            solver.linear_solver_tolerance = levels[0]


def _assert_same_step(actual, expected, name):
    # Bit for bit: the replayed graph issues the launches of the eager step, with the same workspace.
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32), err_msg=name)


if __name__ == "__main__":
    unittest.main()
