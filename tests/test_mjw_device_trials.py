"""Physical articulated trials and rollback, including the conditional outer loop.

The embedded fixture uses accepted geometric probes only to initialize material
history. It is not a procedural insertion benchmark. Subsequent intervals use
real FEM, articulated MuJoCoWarp contacts and unchanged physical acceptance.
"""

import os
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from unittest.mock import PropertyMock, patch

import numpy as np
import test_grasp_scene
import warp as wp
from _warp_step_support import activities
from test_grasp_scene import _scene
from test_needle_shaft_preconditioner import history

from newton_solver_suturebot import NeedleStepTooLarge, NeedleTissueCoupler, SolverFEM
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_device import (
    _padded_samples as woodbury_padded_samples,
)
from newton_solver_suturebot._src.solvers.coupled.mujoco_warp_trials import CapturedArticulatedTrials, _include_overflow
from newton_solver_suturebot._src.solvers.fem.certified_cg import CertifiedCorrection
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco_warp import INTEGRATION_FIELDS
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene

FIELDS = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
DT = 0.0001


def _embedded(scene, capture, *, surface=False):
    probe = scene.model.state()
    probe.assign(scene.state_0)
    # The six-tet fixture has bottom-supported cells; stay in the shallow
    # admissible part of the tunnel instead of approaching their fixed nodes.
    final = 0.1 * scene.plan.exit_angle
    count = int(np.ceil((final - scene.plan.initial_angle) * scene.geometry.radius / 0.00005))
    alm = scene.interaction.tunnel_alm
    needle_com = scene.model.body_com.numpy()[scene.needle_body]
    for angle in np.linspace(scene.plan.initial_angle, final, count + 1):
        poses = probe.body_q.numpy().copy()
        poses[scene.needle_body] = scene.plan.pose(angle)
        probe.body_q.assign(poses)
        alm.begin_step()
        probe.clear_forces()
        scene._external_forces.zero_()
        scene.interaction.add_forces(probe, scene.needle_body, needle_com, scene._external_forces, DT, implicit=True)
        if not alm.converged(scene.args.coupling_absolute_tolerance, scene.args.coupling_relative_tolerance):
            raise AssertionError("Initial geometric probe must have a converged tunnel reaction")
        scene.interaction.accept(probe, scene.needle_body)
        if scene.interaction.status == "failed":
            raise RuntimeError(scene.interaction.reason)
        alm.commit()
    # Rebuild the physical tool reset pose once, after geometric history seeding.
    # No simulated timestep has advanced and there is no needle weld or drive.
    scene._initialize_pose(scene.plan.pose(final))
    scene.cartesian_target = scene.state_0.body_q.numpy()[scene.graspers[0].palm_body].astype(float)
    scene.state_0.clear_forces()
    scene._prepare_step_loads(scene.cartesian_target, np.zeros(6), DT)
    scene.solver.refresh_tangent(scene.state_0.particle_q, DT)
    if scene.coupler is not None:
        # The coupler that the scene was built with may have captured its graphs at start-up. The
        # pose reset above replaced what they bind (rigid solver, held start, certificate scope,
        # response basis), and this fixture replaces the coupler: its graphs are dropped first,
        # which also releases the Krylov workspaces they pinned in the solver's registry.
        scene.coupler.invalidate_fem_graph()
    scene.coupler = NeedleTissueCoupler(
        scene.model,
        scene.solver,
        scene.rigid_solver,
        scene.interaction,
        scene.needle_body,
        scene.model.body_com.numpy()[scene.needle_body],
        moment_length=scene.geometry.radius,
        capture_fem=True,
        capture_coupling=capture,
        shaft_preconditioner=True,
        # With the surface metric the correction is the combined one, as in the grasp scene.
        surface_preconditioner=surface,
        preconditioned_anderson=surface,
        relaxation=1.0,
        max_iterations=160,
        absolute_tolerance=scene.args.coupling_absolute_tolerance,
        relative_tolerance=scene.args.coupling_relative_tolerance,
    )
    scene.coupler._gpu_trial_policy = "device-aitken" if capture else "current"
    if not scene.coupler._shaft_preconditioner.active:
        raise AssertionError("Embedded fixture must exercise the established, preconditioned branch")
    # One real structural solve prepares standard scratch before graph lookup.
    # Preserve the physical output and numerical guess owned by the transaction.
    increment = wp.clone(scene.solver.velocity_increment)
    forces = wp.clone(scene.state_0.particle_f)
    temporary = scene.model.state()
    scene.solver.step(scene.state_0, temporary, scene.control, None, DT)
    wp.copy(scene.solver.velocity_increment, increment)
    wp.copy(scene.state_0.particle_f, forces)


@contextmanager
def _pair(*, reference_capture=False):
    make_config = test_grasp_scene.SutureSceneConfig
    with (
        patch.object(
            test_grasp_scene,
            "SutureSceneConfig",
            side_effect=lambda **kwargs: replace(make_config(**kwargs), rigid_backend="mujoco-warp"),
        ),
        ExitStack() as stack,
    ):
        reference, _ = stack.enter_context(_scene())
        captured, _ = stack.enter_context(_scene())
        _embedded(reference, reference_capture)
        _embedded(captured, True)
        yield reference, captured


@contextmanager
def _surface_scene():
    """One embedded scene on the MuJoCo Warp backend whose captured trials hold the surface metric."""
    make_config = test_grasp_scene.SutureSceneConfig
    with (
        patch.object(
            test_grasp_scene,
            "SutureSceneConfig",
            side_effect=lambda **kwargs: replace(make_config(**kwargs), rigid_backend="mujoco-warp"),
        ),
        _scene() as (scene, _),
    ):
        _embedded(scene, True, surface=True)
        scene.coupler._gpu_trial_policy = "device-anderson"
        yield scene


@contextmanager
def _surface_pair():
    """Two such scenes, built alike."""
    make_config = test_grasp_scene.SutureSceneConfig
    with (
        patch.object(
            test_grasp_scene,
            "SutureSceneConfig",
            side_effect=lambda **kwargs: replace(make_config(**kwargs), rigid_backend="mujoco-warp"),
        ),
        ExitStack() as stack,
    ):
        scenes = [stack.enter_context(_scene())[0] for _ in range(2)]
        for scene in scenes:
            _embedded(scene, True, surface=True)
            scene.coupler._gpu_trial_policy = "device-anderson"
        yield scenes


def _eager_flag(interaction):
    """The flag of the shaft correction as its begin_step evaluated it: a shaft law and an engaged length."""
    count = interaction.track_count
    lengths, engaged = interaction._track_lengths.numpy()[:count], interaction._track_engaged.numpy()[:count]
    law = (
        interaction.tunnel_alm is not None
        or interaction.shaft_stiffness_per_m > 0
        or interaction.shaft_damping_per_m > 0
    )
    return bool(law and float(np.sum(lengths[engaged != 0])) > 0)


@contextmanager
def _engagement_reads(interaction):
    """Count the queries of the engaged length, its device reads and the requests for it."""
    queries = []
    original = type(interaction).engaged_track_length

    def counted(self):
        queries.append(1)
        return original.fget(self)

    with (
        patch.object(type(interaction), "engaged_track_length", property(counted)),
        patch.object(interaction, "_read_engaged_track_length", wraps=interaction._read_engaged_track_length) as read,
        patch.object(
            interaction, "request_engaged_track_length", wraps=interaction.request_engaged_track_length
        ) as request,
    ):
        yield lambda: (len(queries), read.call_count, request.call_count)


def _state(scene):
    return {
        "input": {name: getattr(scene.state_0, name).numpy().copy() for name in FIELDS},
        "output": {name: getattr(scene.state_1, name).numpy().copy() for name in FIELDS},
        "integration": {name: getattr(scene.rigid_solver.mjw_data, name).numpy().copy() for name in INTEGRATION_FIELDS},
        "metrics": scene.rigid_solver.accepted_grasp_metrics.numpy().copy(),
        "increment": scene.solver.velocity_increment.numpy().copy(),
        "load": scene.coupler._load.numpy().copy(),
        "history": history(scene.interaction),
    }


class OverflowFlagTests(unittest.TestCase):
    def test_any_rigid_overflow_invalidates_a_candidate_without_host_readback(self):
        overflow = wp.array([0, 0, 4, 0], dtype=int, device="cpu")
        valid = wp.ones(1, dtype=int, device="cpu")
        wp.launch(_include_overflow, 4, [overflow, valid], device="cpu")
        np.testing.assert_array_equal(valid.numpy(), [0])
        overflow.zero_()
        valid.fill_(1)
        wp.launch(_include_overflow, 4, [overflow, valid], device="cpu")
        np.testing.assert_array_equal(valid.numpy(), [1])


class EmbeddedFixtureTests(unittest.TestCase):
    def test_geometric_history_initialization_preserves_zero_physical_time(self):
        # Exercise fixture preparation on CPU with the native backend, before
        # spending a GPU validation run on graph execution.
        with patch.dict(os.environ, {"NEWTON_TEST_DEVICE": "cpu", "SUTUREBOT_EXPERIMENTAL_MJW": "0"}):
            with _scene() as (scene, _):
                _embedded(scene, False)
                self.assertEqual(scene.elapsed, 0)
                self.assertGreater(scene.interaction.track_count, 1)
                self.assertTrue(scene.coupler._shaft_preconditioner.active)
                self.assertFalse(scene.interaction.tunnel_alm._open)


class ArticulatedDeviceTrialTests(unittest.TestCase):
    def setUp(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("Articulated conditional coupling requires CUDA")

    def _step(self, scene, dt=DT):
        scene.coupler.step(scene.state_0, scene.state_1, scene.control, dt, external_forces=scene._external_forces)

    def test_established_gpu_outer_graph_matches_gpu_host_trials_and_advances_time_once(self):
        with _pair() as (reference, captured):
            # The device and host loops visit different trial sequences. Equal
            # endpoints at this tolerance need rigid solves that do not depend on
            # the previous trial, so compare the two loops without trial warm starts.
            for scene in (reference, captured):
                scene.rigid_solver.trial_warm_start = False
            # Leaving the current puncture does not remove an engaged material
            # track. Selection must follow the actual host acceleration branch,
            # which is preconditioned Aitken whenever this metric is active.
            captured.coupler._gpu_checkpoint.save()
            with patch.object(captured.interaction, "punctured", False):
                self.assertIsNotNone(captured.coupler._device_trials(captured.state_0, captured.control, DT))
            for scene in (reference, captured):
                initial_time = scene.rigid_solver.mjw_data.time.numpy()[0]
                self._step(scene)
                expected = initial_time
                for _ in range(4):
                    expected = np.float32(expected + np.float32(DT / 4))
                np.testing.assert_array_equal(scene.rigid_solver.mjw_data.time.numpy(), [expected])
                scene.rigid_solver.check_overflow()
                self.assertTrue(np.isfinite(scene.state_1.particle_q.numpy()).all())
                np.testing.assert_array_equal(
                    scene.rigid_solver.mjw_model.opt.tolerance.numpy(), np.array([1e-6], dtype=np.float32)
                )
            self.assertGreater(captured.coupler.coupling_graph_launches, 0)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            self.assertEqual(reference.coupler.coupling_graph_launches, 0)
            self.assertEqual(history(reference.interaction), history(captured.interaction))
            for name in FIELDS:
                np.testing.assert_allclose(
                    getattr(captured.state_1, name).numpy(),
                    getattr(reference.state_1, name).numpy(),
                    rtol=3e-5,
                    atol=2e-7,
                    err_msg=name,
                )
            np.testing.assert_array_equal(
                captured.state_1.particle_q.numpy()[captured.fixed_vertex_indices],
                captured.state_0.particle_q.numpy()[captured.fixed_vertex_indices],
            )

    def test_trial_warm_starts_keep_endpoints_within_the_rigid_solver_tolerance(self):
        import mujoco

        with _pair() as (reference, captured):
            captured.coupler._gpu_checkpoint.save()
            with patch.object(captured.interaction, "punctured", False):
                self.assertIsNotNone(captured.coupler._device_trials(captured.state_0, captured.control, DT))
            for scene in (reference, captured):
                self.assertTrue(scene.rigid_solver.trial_warm_start)
                self._step(scene)
                scene.rigid_solver.check_overflow()
                # The accepted trial started every microstep from a saved solution.
                self.assertEqual(int(scene.rigid_solver._trial_warm_valid.numpy()[0]), 1)
            self.assertGreater(captured.coupler.coupling_graph_launches, 0)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            self.assertEqual(reference.coupler.coupling_graph_launches, 0)
            self.assertEqual(history(reference.interaction), history(captured.interaction))
            for name in ("particle_q", "particle_qd", "joint_q"):
                np.testing.assert_allclose(
                    getattr(captured.state_1, name).numpy(),
                    getattr(reference.state_1, name).numpy(),
                    rtol=3e-5,
                    atol=2e-7,
                    err_msg=name,
                )
            # A warm-started solve may stop at a different point inside the
            # solver's tolerance. The solver stops once its cost is within
            # tolerance * trace(M) of the minimum, and the cost is at least
            # 0.5 * |a - a*|^2 in the M norm, so each acceleration component is
            # within sqrt(2 * tolerance * trace(M) * inv(M)_ii) of the solution.
            # Velocities integrate that over the interval; two runs double it.
            solver = captured.rigid_solver
            solver.sync_native_shadow(forward=True)
            model = solver.mj_model
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, solver.mj_data, mass)
            gap = solver.gpu_tolerance * np.trace(mass)
            bound = 2 * DT * np.sqrt(2 * gap * np.diag(np.linalg.inv(mass)))
            difference = np.abs(
                captured.rigid_solver.mjw_data.qvel.numpy()[0] - reference.rigid_solver.mjw_data.qvel.numpy()[0]
            )
            self.assertTrue(np.all(difference <= bound), msg=f"difference {difference} exceeds {bound}")

    def test_device_anderson_runs_inside_the_actual_coupled_graph_and_resets_after_rejection(self):
        with _pair() as (_, captured):
            captured.coupler._gpu_trial_policy = "device-anderson"
            load = captured.coupler._load.numpy().copy()
            load[-6] += 1e-4
            captured.coupler._load.assign(load)
            before = _state(captured)
            with patch.object(
                captured.coupler._tet_step_guard, "check", side_effect=NeedleStepTooLarge("late AA guard")
            ):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late AA guard"):
                    self._step(captured)
            graph = captured.coupler._captured_trials
            self.assertIsNotNone(graph.rigid.anderson)
            self.assertGreater(int(graph.rigid.anderson.counters.numpy()[0]), 0)
            np.testing.assert_array_equal(captured.coupler._load.numpy(), before["load"])
            for field, expected in before["integration"].items():
                np.testing.assert_array_equal(getattr(captured.rigid_solver.mjw_data, field).numpy(), expected)
            self._step(captured)
            self.assertGreaterEqual(captured.coupler.coupling_graph_launches, 2)
            self.assertTrue(captured.rigid_solver.diagnostics()["gpu_accepted_clock_matches"])
            self.assertFalse(captured.interaction.tunnel_alm._open)

    def test_contact_preconditioner_guess_is_solved_on_device_inside_the_coupled_graph(self):
        with _pair() as (_, captured):
            captured.coupler._gpu_trial_policy = "device-anderson"
            load = captured.coupler._load.numpy().copy()
            load[-6] += 1e-4
            captured.coupler._load.assign(load)
            self._step(captured)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            rigid = captured.coupler._captured_trials.rigid
            guess = captured.coupler._shaft_preconditioner._woodbury_guess
            self.assertIs(rigid.guess, guess)
            self.assertGreater(int(rigid.anderson.counters.numpy()[0]), 0)
            # The response basis is bound on the host; the packet readback and
            # dense host solve of the native path never ran for these corrections.
            self.assertEqual((guess.preparations, guess.ready, guess.fallbacks), (0, 0, 0))
            np.testing.assert_array_equal(guess.enabled.numpy(), [1])
            counters = guess.diagnostics()["counters"]
            # One warm application on a zero residual precedes the corrections.
            self.assertGreater(counters["applications"], 1)
            self.assertGreater(counters["selected"], 0)
            self.assertEqual(counters["cg_cap_hits"], 0)
            self.assertFalse(captured.interaction.tunnel_alm._open)

    def test_zero_contact_preconditioner_start_remains_selectable(self):
        with _pair() as (_, captured):
            captured.coupler._gpu_trial_policy = "device-anderson"
            captured.coupler._gpu_contact_guess = False
            load = captured.coupler._load.numpy().copy()
            load[-6] += 1e-4
            captured.coupler._load.assign(load)
            self._step(captured)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            rigid = captured.coupler._captured_trials.rigid
            self.assertGreater(int(rigid.anderson.counters.numpy()[0]), 0)
            self.assertIsNone(rigid.guess)
            self.assertIsNone(rigid.guess_key)
            # No host-controlled correction ran, so nothing created a guess.
            self.assertIsNone(captured.coupler._shaft_preconditioner._woodbury_guess)
            self.assertIsNone(captured.coupler._shaft_preconditioner._guess_storage)
            self.assertFalse(captured.interaction.tunnel_alm._open)

    def test_guess_and_its_key_do_not_follow_the_track_and_new_samples_capture_nothing(self):
        from test_shaft_preconditioner_storage import append_released_sample

        with _pair() as (_, captured):
            coupler = captured.coupler
            pc = coupler._shaft_preconditioner
            captured.coupler._gpu_trial_policy = "device-anderson"

            def step():
                # A carried load that the first trial does not balance, so that the trials need a correction.
                load = coupler._load.numpy().copy()
                load[-6] += 1e-4
                coupler._load.assign(load)
                self._step(captured)
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                captured.state_0, captured.state_1 = captured.state_1, captured.state_0

            # One capture for each of the two state buffers.
            step()
            step()
            self.assertEqual(coupler.coupling_graph_captures, 2)
            trials = coupler._captured_trials
            rigid = trials.rigid
            guess, key, signature = rigid.guess, rigid.guess_key, trials.signature
            self.assertIsNotNone(key)
            self.assertIs(guess, pc._guess_storage)
            self.assertIs(pc._woodbury_guess, guess)
            # The key of the guess is the entry before the last one, which is the epoch of the
            # FEM solver's registry of Krylov workspaces.
            self.assertEqual(signature[-2], key)
            self.assertEqual(signature[-1], captured.solver.linear_binding_epoch())
            self.assertEqual(guess.available.numpy().tolist(), [1])
            solve = guess.device_solve
            pointers = [array.ptr for array in guess.owners + solve.owners]
            applications = guess.diagnostics()["counters"]["applications"]
            self.assertGreater(applications, 0)

            def other_captures():
                # The graphs beside the trial graph that address the track: the solve graph of the
                # correction itself, the preparation of its inputs and the preparation of the host
                # packet of the guess.
                inputs = getattr(pc._input_preparation, "replay", None)
                packet = getattr(guess._packet_preparation, "replay", None)
                return (pc.graph_captures, getattr(inputs, "captures", 0), getattr(packet, "captures", 0))

            others = other_captures()
            # At most the two forms of the solve graph and one capture of each preparation.
            self.assertLessEqual(others[0], 2)
            self.assertLessEqual(max(others[1:]), 1)
            # New track samples, one at a time, each followed by a step from either state buffer.
            grown = 0
            for _ in range(3):
                try:
                    count = append_released_sample(captured)
                except StopIteration:
                    break
                grown += 1
                step()
                step()
                self.assertEqual(pc.count, count)
                self.assertIs(rigid.guess, guess)
                self.assertIs(guess.device_solve, solve)
                self.assertEqual(rigid.guess_key, key)
                current = trials.signature
                changed = [i for i, (old, new) in enumerate(zip(signature, current, strict=True)) if old != new]
                self.assertEqual(changed, [])
                self.assertEqual(coupler.coupling_graph_captures, 2)
                self.assertEqual(other_captures(), others)
                self.assertEqual([array.ptr for array in guess.owners + solve.owners], pointers)
                self.assertEqual(guess.available.numpy().tolist(), [1])
                self.assertEqual(solve.capacity, woodbury_padded_samples(count))
                later = guess.diagnostics()["counters"]["applications"]
                self.assertGreater(later, applications)
                applications = later
            self.assertGreater(grown, 0)
            # The binding of the captured loops read no packet and solved nothing on the host.
            self.assertEqual((guess.preparations, guess.ready, guess.fallbacks), (0, 0, 0))
            # Launches that may not use the basis (a track beyond the scratch of the guess): the same
            # graphs, the zero start, nothing counted.
            from newton_solver_suturebot._src.solvers.coupled.acceleration import woodbury_guess

            self.assertGreater(pc.count, 1)
            with patch.object(woodbury_guess, "_MAX_TRACKS", 1):
                step()
                step()
                self.assertEqual(guess.available.numpy().tolist(), [0])
            self.assertEqual(rigid.guess_key, key)
            self.assertEqual(trials.signature, signature)
            self.assertEqual(coupler.coupling_graph_captures, 2)
            self.assertEqual(guess.diagnostics()["counters"]["applications"], applications)
            step()
            self.assertEqual(guess.available.numpy().tolist(), [1])
            self.assertEqual(coupler.coupling_graph_captures, 2)
            self.assertGreater(guess.diagnostics()["counters"]["applications"], applications)
            # Neither the samples nor the launches without a usable basis captured another graph.
            self.assertEqual(other_captures(), others)
            self.assertFalse(captured.interaction.tunnel_alm._open)

    def test_increment_guess_only_moves_the_first_trial_load_and_never_survives_rejection_or_invalidation(self):
        from newton_solver_suturebot._src.solvers.coupled.phase_predictor import PhaseLoadPredictor, coupler_signature

        with _pair() as (_, captured):
            coupler = captured.coupler
            coupler._gpu_trial_policy = "device-anderson"
            self.assertIsNotNone(coupler_signature(coupler, increments=True))
            predictor = coupler._phase_load_predictor = PhaseLoadPredictor(coupler._load, increments=True)
            command = np.array([0, 0, 0, 0, 0, 0, 1, 0.0])
            accepted, starts = [], []
            solve = coupler._solve_device

            def recorded(*args):
                # The first trial load of the captured loop, read before its graph is launched.
                starts.append(coupler._trial_load.numpy().copy())
                return solve(*args)

            def frame(index, *, success=True):
                # One control frame of one physical step.
                accepted.append(coupler._load.numpy().copy())
                predictor.begin_frame(frame_dt=DT, step_dt=DT, elapsed=index * DT, command=command, reset_token=0)
                try:
                    self._step(captured)
                    captured.state_0, captured.state_1 = captured.state_1, captured.state_0
                finally:
                    predictor.end_frame((index + 1) * DT if success else index * DT, success)

            def record_agreeing_changes():
                # The last two frames changed one rigid force row by the same amount.
                values = np.zeros(coupler._load.size)
                values[-6] = 1e-4
                for record in predictor.frames[:2]:
                    record.increments[0].assign(values)
                return values

            with patch.object(coupler, "_solve_device", recorded):
                frame(0)
                frame(1)
                self.assertEqual((predictor.depth, predictor.attempts), (2, 0))
                np.testing.assert_array_equal(starts, accepted)
                change = record_agreeing_changes()
                frame(2)
                np.testing.assert_array_equal(starts[2], accepted[2] + change)
                np.testing.assert_array_equal(predictor.counts.numpy(), [0, 0, 0, 1])
                # The guessed step ran in the captured loop and passed the unchanged acceptance.
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                self.assertEqual(coupler.coupling_graph_launches, 3)
                self.assertGreater(coupler.iterations, 1)
                self.assertLess(abs(coupler._load.numpy()[-6] - accepted[2][-6]), 1e-4)

                # A step rejected after its completed graph keeps neither its guess nor the history.
                change = record_agreeing_changes()
                with patch.object(coupler._tet_step_guard, "check", side_effect=NeedleStepTooLarge("late guess guard")):
                    with self.assertRaisesRegex(NeedleStepTooLarge, "late guess guard"):
                        frame(3, success=False)
                np.testing.assert_array_equal(starts[3], accepted[3] + change)
                np.testing.assert_array_equal(coupler._load.numpy(), accepted[3])
                self.assertEqual((predictor.depth, predictor.open, predictor.frame_open), (0, False, False))
                # The rejected guessed step is counted and is not repeated from another start.
                self.assertEqual((len(starts), predictor.rejections), (4, 1))
                frame(3)
                frame(4)
                np.testing.assert_array_equal(starts[4:], accepted[4:])
                self.assertEqual((predictor.depth, predictor.attempts), (2, 2))

                # Discarding the captured graphs between steps discards the recorded changes with them.
                record_agreeing_changes()
                coupler.invalidate_fem_graph()
                self.assertEqual(predictor.depth, 0)
                frame(5)
                np.testing.assert_array_equal(starts[6], accepted[6])
                self.assertEqual((predictor.depth, predictor.attempts), (1, 2))

    def test_guessed_step_that_exhausts_its_trial_budget_is_rejected_once_and_restarts_from_the_accepted_load(self):
        from newton_solver_suturebot import NeedleCouplingNotConverged
        from newton_solver_suturebot._src.solvers.coupled.phase_predictor import PhaseLoadPredictor

        with _pair() as (_, captured):
            coupler = captured.coupler
            coupler._gpu_trial_policy = "device-anderson"
            predictor = coupler._phase_load_predictor = PhaseLoadPredictor(coupler._load, increments=True)
            command = np.array([0, 0, 0, 0, 0, 0, 1, 0.0])
            starts, integration, exhausted = [], [], []
            solve = coupler._solve_device

            def recorded(*args):
                # What the captured loop starts from: its first trial load and the rigid integration state.
                starts.append(coupler._trial_load.numpy().copy())
                integration.append(_state(captured)["integration"])
                result = solve(*args)
                # The completed graph is reported as a trial budget that ended without convergence.
                return False if exhausted and exhausted.pop() else result

            def frame(index, *, success=True):
                predictor.begin_frame(frame_dt=DT, step_dt=DT, elapsed=index * DT, command=command, reset_token=0)
                try:
                    self._step(captured)
                    captured.state_0, captured.state_1 = captured.state_1, captured.state_0
                finally:
                    predictor.end_frame((index + 1) * DT if success else index * DT, success)

            with patch.object(coupler, "_solve_device", recorded):
                frame(0)
                frame(1)
                change = np.zeros(coupler._load.size)
                change[-6] = 1e-4
                for record in predictor.frames[:2]:
                    record.increments[0].assign(change)
                before = _state(captured)
                exhausted.append(True)
                with self.assertRaises(NeedleCouplingNotConverged):
                    frame(2, success=False)
                # One launch from the guessed start: the trial budget of the step is not spent a second time.
                self.assertEqual((len(starts), predictor.attempts, predictor.rejections), (3, 1, 1))
                np.testing.assert_array_equal(starts[2], before["load"] + change)
                # The rejection restored the accepted load and rigid integration state and kept no history.
                np.testing.assert_array_equal(coupler._load.numpy(), before["load"])
                for field, expected in before["integration"].items():
                    np.testing.assert_array_equal(
                        getattr(captured.rigid_solver.mjw_data, field).numpy(), expected, err_msg=field
                    )
                self.assertEqual((predictor.depth, predictor.commits, predictor.open), (0, 2, False))
                self.assertFalse(captured.interaction.tunnel_alm._open)

                # The caller's next attempt of the step starts from the accepted load, as without a predictor.
                frame(2)
                np.testing.assert_array_equal(starts[3], before["load"])
                for field, expected in before["integration"].items():
                    np.testing.assert_array_equal(integration[3][field], expected, err_msg=field)
                self.assertEqual((coupler.coupling_graph_launches, coupler.coupling_graph_fallbacks), (4, 0))
                self.assertTrue(captured.rigid_solver.diagnostics()["gpu_accepted_clock_matches"])
                self.assertEqual((predictor.attempts, predictor.rejections, predictor.depth), (1, 1, 1))

    def test_rejection_after_completed_graph_restores_integration_output_metrics_and_material_history(self):
        with _pair() as (_, captured):
            saved = _state(captured)
            with patch.object(captured.coupler._tet_step_guard, "check", side_effect=NeedleStepTooLarge("late guard")):
                with self.assertRaisesRegex(NeedleStepTooLarge, "late guard"):
                    self._step(captured)
            self.assertGreater(captured.coupler.coupling_graph_launches, 0)
            current = _state(captured)
            for group in ("input", "output", "integration"):
                for name, expected in saved[group].items():
                    np.testing.assert_array_equal(current[group][name], expected, err_msg=f"{group}.{name}")
            for name in ("metrics", "increment", "load"):
                np.testing.assert_array_equal(current[name], saved[name], err_msg=name)
            self.assertEqual(current["history"], saved["history"])
            self._step(captured)
            self.assertEqual(captured.coupler.coupling_graph_launches, 2)

    def test_accepted_buffer_swaps_reuse_both_graph_bindings(self):
        with _pair() as (_, captured):
            track_count = captured.interaction.track_count
            captures = []
            for _ in range(4):
                self._step(captured)
                self.assertEqual(captured.interaction.track_count, track_count)
                self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
                captures.append(captured.coupler.coupling_graph_captures)
                captured.state_0, captured.state_1 = captured.state_1, captured.state_0
            # Two physical state buffers need two captures. Returning to each
            # accepted buffer must reuse its own graph and persistent FEM work.
            self.assertEqual(captures, [1, 2, 2, 2])
            self.assertEqual(captured.coupler.coupling_graph_launches, 4)
            # That work is the Krylov workspaces of the solver's registry, one per tolerance
            # level, and a device solve leaves one of them selected.
            trials, solver = captured.coupler._captured_trials, captured.solver
            self.assertEqual([settings[3] for settings, _ in trials.linear_solvers], solver.tolerance_levels())
            registered = list(solver._linear_workspaces.values())
            for _, workspace in trials.linear_solvers:
                self.assertTrue(any(workspace is value for value in registered))
            self.assertTrue(any(solver._lin_solver is value for value in registered))
            self.assertEqual(trials.signature[-1], solver.linear_binding_epoch())

    def test_start_up_captures_the_trial_graph_of_both_buffers_and_steps_then_only_launch(self):
        # NeedleTissueCoupler.capture_graphs captures and launches the trial graph of each accepted-state
        # buffer inside an attempt that ends like a rejected step. The scene is then as it was found,
        # and the device trials of a step on either buffer launch the graph of that buffer.
        with _pair(reference_capture=True) as (plain, started):
            for scene in (plain, started):
                scene.state_1.assign(scene.state_0)
            coupler = started.coupler
            self.assertIsNone(coupler._captured_trials)
            saved = _state(started)
            work = (
                "iterations",
                "evaluations",
                "coupling_graph_launches",
                "coupling_graph_fallbacks",
                "fem_graph_launches",
            )
            counters = [getattr(coupler, name) for name in work]
            held, rigid = started.solver._held_start, started.rigid_solver
            host = (
                held.preparations,
                held.graph_launches,
                rigid.gpu_graph_launches,
                started.solver.linear_solver_tolerance,
            )
            self.assertTrue(coupler.capture_graphs((started.state_0, started.state_1), started.control, DT))
            current = _state(started)
            for group in ("input", "output", "integration"):
                for name, expected in saved[group].items():
                    np.testing.assert_array_equal(current[group][name], expected, err_msg=f"{group}.{name}")
            for name in ("metrics", "increment", "load"):
                np.testing.assert_array_equal(current[name], saved[name], err_msg=name)
            self.assertEqual(current["history"], saved["history"])
            self.assertEqual([getattr(coupler, name) for name in work], counters)
            self.assertEqual(
                (
                    held.preparations,
                    held.graph_launches,
                    rigid.gpu_graph_launches,
                    started.solver.linear_solver_tolerance,
                ),
                host,
            )
            self.assertFalse(started.interaction.tunnel_alm._open)
            # Two graphs, one per buffer, each launched once; the launches are not counted as work.
            trials = coupler._captured_trials
            self.assertEqual((trials.captures, trials.launches, len(trials.graphs)), (2, 0, 2))
            self.assertEqual(coupler.coupling_graph_captures, 2)
            self.assertEqual([entry[0].graph_exec is None for entry in trials.graphs.values()], [False, False])
            # The fixture has an accepted tip, so its first step is a device step: the accepted
            # positions that start-up committed to the tetrahedron query stay committed.
            self.assertIsNotNone(started.interaction._previous_tip)
            self.assertTrue(started.interaction._device_queries.has_previous)
            signature = trials.signature
            # Steps from either buffer: the scene that started up captures nothing more.
            for step in range(4):
                for scene in (plain, started):
                    self._step(scene)
                    self.assertEqual(scene.coupler.coupling_graph_fallbacks, 0)
                self.assertEqual(coupler.coupling_graph_captures, 2, step)
                self.assertEqual(trials.signature, signature)
                self.assertEqual(coupler.coupling_graph_launches, step + 1)
                self.assertEqual(plain.coupler.coupling_graph_captures, min(step + 1, 2))
                self.assertEqual(history(started.interaction), history(plain.interaction))
                # The accepted states agree within what the start-up call may change. The bound on a
                # velocity is absolute, with no relative part, and is the one of the grasp scene test
                # (tests/test_grasp_scene.py). Measured in this fixture on cuda:0 (2026-11-17, three
                # pairs of scenes): two scenes without the call agree bit for bit in the rigid
                # fields, and the call moves a rigid velocity by 1.07e-5 at most (at a scale of
                # 5.8e-2), the same value in every pair.
                for name in FIELDS:
                    velocity = name.endswith("qd")
                    np.testing.assert_allclose(
                        getattr(started.state_1, name).numpy(),
                        getattr(plain.state_1, name).numpy(),
                        rtol=0 if velocity else 3e-5,
                        atol=2e-5 if velocity else 2e-7,
                        err_msg=f"step {step}: {name}",
                    )
                for scene in (plain, started):
                    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
            self.assertIs(coupler._captured_trials, trials)
            self.assertEqual(len(trials.graphs), 2)
            # A second call finds the graphs of both buffers and captures none.
            self.assertTrue(coupler.capture_graphs((started.state_0, started.state_1), started.control, DT))
            self.assertEqual((trials.captures, trials.launches, coupler.coupling_graph_launches), (2, 4, 4))

    def test_host_trials_after_device_trials_replay_captured_fem_steps(self):
        # The trial graph binds the Krylov workspaces of the FEM solver's registry, one per
        # tolerance level, fetched inside the attempt, where the certified solve is active. The
        # host-controlled trials of a later attempt select the same workspaces, so every FEM
        # step that Python issues for them is the capture of a graph and every other one a
        # replay: no eager step, and no workspace beyond those of the device trials.
        with _pair() as (_, captured):
            coupler, solver = captured.coupler, captured.solver
            for _ in range(2):
                self._step(captured)
                captured.state_0, captured.state_1 = captured.state_1, captured.state_0
            self.assertEqual((coupler.coupling_graph_launches, coupler.coupling_graph_fallbacks), (2, 0))
            trials = coupler._captured_trials
            levels = solver.tolerance_levels()
            bound = [workspace for _, workspace in trials.linear_solvers]
            self.assertEqual([settings[3] for settings, _ in trials.linear_solvers], levels)
            registered = list(solver._linear_workspaces.values())
            for workspace in bound:
                self.assertIsInstance(workspace, CertifiedCorrection)
                self.assertTrue(any(workspace is value for value in registered))
            self.assertTrue(any(solver._lin_solver is value for value in registered))
            epoch = solver.linear_binding_epoch()
            self.assertEqual(trials.signature[-1], epoch)
            original, issued = SolverFEM.step, []

            def counted(self, *args, **kwargs):
                if self is solver:
                    issued.append(bool(self.model.device.is_capturing))
                return original(self, *args, **kwargs)

            captures, launches = coupler.fem_graph_captures, coupler.fem_graph_launches
            coupler.capture_coupling = False
            with (
                patch.object(SolverFEM, "step", counted),
                patch.object(solver, "_make_linear_solver", wraps=solver._make_linear_solver) as allocate,
            ):
                for _ in range(4):
                    self._step(captured)
                    self.assertEqual(coupler.fem_graph_status, "replaying")
                    self.assertTrue(any(solver._lin_solver is workspace for workspace in bound))
                    captured.state_0, captured.state_1 = captured.state_1, captured.state_0
                self.assertEqual(allocate.call_count, 0)
            self.assertEqual(coupler.coupling_graph_launches, 2)
            self.assertGreater(len(issued), 0)
            self.assertTrue(all(issued))
            self.assertEqual(len(issued), coupler.fem_graph_captures - captures)
            self.assertGreaterEqual(coupler.fem_graph_launches - launches, 4)
            self.assertEqual(solver.linear_binding_epoch(), epoch)
            # Device trials again, with the workspaces they bound before the host-controlled steps.
            coupler.capture_coupling = True
            self._step(captured)
            self.assertEqual((coupler.coupling_graph_launches, coupler.coupling_graph_fallbacks), (3, 0))
            for kept, (_, workspace) in zip(bound, trials.linear_solvers, strict=True):
                self.assertIs(workspace, kept)
            self.assertEqual(trials.signature[-1], epoch)

    def test_trial_graph_binds_the_workspaces_of_the_current_certificate_state(self):
        # The solver's registry keys a workspace by the state of the certificate scope: in an
        # attempt that the scope supports the solve is the certified one, otherwise the plain
        # one. The trial graph binds the workspaces of the attempt it is prepared in and
        # releases the others, which stay registered for the next attempt of their kind.
        with _pair() as (_, captured):
            coupler, solver = captured.coupler, captured.solver
            scope = solver._certified_linear
            self._step(captured)
            captured.state_0, captured.state_1 = captured.state_1, captured.state_0
            trials = coupler._captured_trials
            certified = [workspace for _, workspace in trials.linear_solvers]
            registered = list(solver._linear_workspaces.values())
            for workspace in certified:
                self.assertIsInstance(workspace, CertifiedCorrection)
                self.assertTrue(any(workspace is value for value in registered))
            self.assertEqual(len(solver._pinned_workspaces), len(certified))
            # One attempt whose forward configuration the scope does not support.
            with patch.object(scope, "forward_supported", return_value=False):
                self._step(captured)
            captured.state_0, captured.state_1 = captured.state_1, captured.state_0
            plain = [workspace for _, workspace in trials.linear_solvers]
            self.assertEqual(len(plain), len(certified))
            registered = list(solver._linear_workspaces.values())
            for workspace in plain:
                self.assertNotIsInstance(workspace, CertifiedCorrection)
                self.assertTrue(any(workspace is value for value in registered))
            for workspace in certified:
                self.assertTrue(any(workspace is value for value in registered))
            self.assertEqual(len(solver._pinned_workspaces), len(plain))
            # The next supported attempt binds the certified workspaces again, the ones made before.
            self._step(captured)
            self.assertEqual(coupler.coupling_graph_launches, 3)
            for kept, (_, workspace) in zip(certified, trials.linear_solvers, strict=True):
                self.assertIs(workspace, kept)
            self.assertEqual(len(solver._pinned_workspaces), len(certified))

    def test_timestep_changes_replay_both_graph_bindings_without_recapture(self):
        # dt, dt/2, dt again, then on: each buffer's graph also runs at a time step other than
        # that of its capture. The time step is device data (StepTime), so after one capture
        # per state buffer nothing is captured again. The reference is a second coupler whose
        # graphs are dropped before every step: each of its graphs only ever ran at the time
        # step it was captured with.
        with _pair(reference_capture=True) as (fresh, kept):
            coupler, solver = kept.coupler, kept.rigid_solver
            track_count = kept.interaction.track_count
            graphs = None
            for index, dt in enumerate((DT, DT / 2, DT, DT, DT / 2)):
                microstep = np.float32(dt / solver.rigid_substeps)
                for scene in (fresh, kept):
                    if scene is fresh and scene.coupler._captured_trials is not None:
                        scene.coupler._captured_trials.invalidate()
                    scene.solver.refresh_tangent(scene.state_0.particle_q, dt)
                    clock = scene.rigid_solver.mjw_data.time.numpy()[0]
                    self._step(scene, dt)
                    # The rigid interval ran its four microsteps at exactly this dt / 4.
                    for _ in range(solver.rigid_substeps):
                        clock = np.float32(clock + microstep)
                    np.testing.assert_array_equal(scene.rigid_solver.mjw_data.time.numpy(), [clock])
                    np.testing.assert_array_equal(scene.rigid_solver.mjw_model.opt.timestep.numpy(), [microstep])
                    self.assertTrue(scene.rigid_solver.diagnostics()["gpu_accepted_clock_matches"])
                    self.assertEqual(scene.interaction.track_count, track_count)
                    self.assertEqual(scene.coupler.coupling_graph_fallbacks, 0)
                    self.assertEqual(scene.coupler.coupling_graph_launches, index + 1)
                # One capture per state buffer for the kept graphs, one per step for the reference.
                self.assertEqual(coupler.coupling_graph_captures, min(index + 1, 2))
                self.assertEqual(fresh.coupler.coupling_graph_captures, index + 1)
                # One object holds the time step of the coupled step for every stage.
                self.assertEqual(coupler.step_time.value, dt)
                for owner in (kept.solver, kept.interaction, solver, coupler._shaft_preconditioner):
                    self.assertIs(owner.step_time, coupler.step_time)
                self.assertIs(coupler._captured_trials.rigid.metric.step_time, coupler.step_time)
                np.testing.assert_array_equal(coupler.step_time.seconds.numpy(), [np.float32(dt)])
                current = [entry[0] for entry in coupler._captured_trials.graphs.values()]
                if index == 1:
                    graphs = current
                    self.assertEqual(len(graphs), 2)
                elif index > 1:
                    self.assertEqual(len(current), 2)
                    for retained, graph in zip(graphs, current, strict=True):
                        self.assertIs(graph, retained)
                self.assertEqual(history(fresh.interaction), history(kept.interaction))
                for name in FIELDS:
                    np.testing.assert_allclose(
                        getattr(kept.state_1, name).numpy(),
                        getattr(fresh.state_1, name).numpy(),
                        rtol=3e-5,
                        atol=2e-7,
                        err_msg=f"step {index} {name}",
                    )
                for scene in (fresh, kept):
                    scene.state_0, scene.state_1 = scene.state_1, scene.state_0

    def test_timestep_changes_replay_the_rigid_solvers_own_graph_without_recapture(self):
        # The rigid interval of the host-controlled trials runs in the rigid solver's own graph
        # (SubcycledMuJoCoWarp.step). A graph stands for the states and the controls of the
        # interval, not for its length: every replay copies dt / rigid_substeps from the step
        # time into MuJoCo Warp's opt.timestep. One capture therefore serves the base step, a
        # halving and a frame remainder, and each replay equals the same interval issued eagerly.
        with _pair() as (_, scene):
            solver, checkpoint = scene.rigid_solver, scene.coupler._gpu_checkpoint
            # Every interval starts from the same integration state and solves unassisted.
            solver.trial_warm_start = False
            self.assertFalse(solver._rigid_warmed)
            replayed, issued = scene.model.state(), scene.model.state()
            names = ("joint_q", "joint_qd", "body_q", "body_qd")
            checkpoint.save()
            start = solver.mjw_data.time.numpy()[0]
            graph, velocities = None, {}
            for index, dt in enumerate((DT, DT, DT / 2, DT, 0.37 * DT, DT / 2, DT)):
                microstep = np.float32(dt / solver.rigid_substeps)
                clock = start
                for _ in range(solver.rigid_substeps):
                    clock = np.float32(clock + microstep)
                endpoints = []
                for output, step in ((replayed, solver.step), (issued, solver._step_uncaptured)):
                    checkpoint.restore()
                    step(scene.state_0, output, scene.control, None, dt)
                    # Four microsteps of exactly this dt / 4, whatever the graph was captured with.
                    np.testing.assert_array_equal(solver.mjw_data.time.numpy(), [clock])
                    np.testing.assert_array_equal(solver.mjw_model.opt.timestep.numpy(), [microstep])
                    solver.check_overflow()
                    endpoints.append({name: getattr(output, name).numpy().copy() for name in names})
                # The first interval warms the rigid step eagerly, the second is captured, the
                # others replay that graph.
                self.assertEqual((solver.gpu_graph_captures, solver.gpu_graph_launches), (min(index, 1), index))
                self.assertEqual(solver.step_time.value, dt)
                self.assertIs(solver.step_time, scene.coupler.step_time)
                current = [entry[0] for entry in solver._rigid_graphs.values()]
                self.assertEqual(len(current), min(index, 1))
                if index == 1:
                    graph = current[0]
                elif index > 1:
                    self.assertIs(current[0], graph)
                for name in names:
                    np.testing.assert_allclose(
                        endpoints[0][name], endpoints[1][name], rtol=3e-5, atol=2e-7, err_msg=f"interval {index} {name}"
                    )
                velocities.setdefault(dt, endpoints[0]["joint_qd"])
            checkpoint.restore()
            # The interval did matter: a graph that kept the microstep of its capture would have
            # ended its replays at the velocities of the base step.
            for dt in (DT / 2, 0.37 * DT):
                self.assertFalse(np.allclose(velocities[dt], velocities[DT], rtol=0.05, atol=0))

    def test_replacing_rigid_model_storage_rebuilds_the_rigid_steps_and_the_graph(self):
        with _pair() as (_, captured):
            self._step(captured)
            solver = captured.rigid_solver
            before = captured.coupler.coupling_graph_captures
            steps = (solver.rigid_step, solver._first_step)
            self.assertNotIn(None, steps)
            self._step(captured)
            # an unchanged model and data keep the graph and the steps
            self.assertEqual(captured.coupler.coupling_graph_captures, before)
            self.assertEqual((solver.rigid_step, solver._first_step), steps)
            solver.mjw_model.geom_friction = wp.clone(solver.mjw_model.geom_friction)
            self._step(captured)
            self.assertGreater(captured.coupler.coupling_graph_captures, before)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)
            self.assertIsNot(solver.rigid_step, steps[0])
            self.assertIsNot(solver._first_step, steps[1])

    def test_trials_of_one_interval_share_the_state_stages_of_their_first_microstep(self):
        with _pair() as (_, captured):
            self._step(captured)
            solver, checkpoint = captured.rigid_solver, captured.coupler._gpu_checkpoint
            rigid = captured.coupler._device_trials(captured.state_0, captured.control, DT).rigid
            # Repetitions are compared with one another, so each must start its solves alike.
            solver.trial_warm_start = False
            checkpoint.save()
            whole, shared = captured.model.state(), captured.model.state()

            def interval(output):
                rigid.step(captured.state_0, output, captured.control, DT)

            # An interval on its own repeats every stage in each of its four microsteps.
            kernels, _ = activities(lambda: interval(whole))
            self.assertEqual((kernels["kinematics_branch"], kernels["next_position"]), (4, 4))
            self.assertEqual(int(solver.mjw_data.overflow.numpy().max()), 0)
            try:
                kernels, _ = activities(lambda: rigid.begin_trials(captured.state_0, captured.control, DT))
                self.assertEqual((kernels["kinematics_branch"], kernels["next_position"]), (1, 0))
                for trial in range(2):
                    kernels, _ = activities(lambda: interval(shared))
                    with self.subTest(trial=trial):
                        # the first microstep keeps what begin_trials issued
                        self.assertEqual((kernels["kinematics_branch"], kernels["next_position"]), (3, 4))
                        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                            np.testing.assert_allclose(
                                getattr(shared, name).numpy(),
                                getattr(whole, name).numpy(),
                                rtol=3e-5,
                                atol=2e-7,
                                err_msg=name,
                            )
                # A flag that the shared stages raised stays with every trial of the interval.
                solver._prepared_overflow.fill_(4)
                interval(shared)
                self.assertEqual(int(solver.mjw_data.overflow.numpy().max()) & 4, 4)
            finally:
                rigid.end_trials()
            interval(shared)
            self.assertEqual(int(solver.mjw_data.overflow.numpy().max()), 0)
            checkpoint.restore()

    def _perturbed_step(self, scene):
        # A carried load that the first trial does not balance, so that the trials need a correction.
        load = scene.coupler._load.numpy().copy()
        load[-6] += 1e-4
        scene.coupler._load.assign(load)
        self._step(scene)
        scene.state_0, scene.state_1 = scene.state_1, scene.state_0

    def test_trials_with_a_surface_metric_never_evaluate_the_flag_of_the_shaft_correction(self):
        # With the surface metric every trial is corrected, by one solve whose launches over the
        # track read the live count: the graph is the same whether the track transmits a shaft
        # interaction or not. The flag is then no entry of the trial signature and nothing in a
        # device step evaluates it, so no step reads the engaged length from the device.
        with _surface_scene() as scene:
            coupler, interaction = scene.coupler, scene.interaction
            pc = coupler._shaft_preconditioner
            self.assertIsNotNone(coupler._surface_preconditioner)
            self.assertIs(pc.surface, coupler._surface_preconditioner)
            self.assertGreater(interaction.track_count, 1)
            # The fixture read the flag when it was built, which asked the first acceptance for the length.
            self._perturbed_step(scene)
            self.assertFalse(interaction._engaged_length_requested)
            with _engagement_reads(interaction) as counts:
                for _ in range(5):
                    self._perturbed_step(scene)
                    self.assertIsNone(pc._active)
                self.assertEqual(counts(), (0, 0, 0))
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                self.assertEqual((coupler.coupling_graph_captures, coupler.coupling_graph_launches), (2, 6))
                trials = coupler._captured_trials
                self.assertTrue(trials.rigid.surface_preconditioned)
                signature = trials.signature
                entry = signature.index(pc.storage_key()) + 1
                self.assertIsNone(signature[entry])
                # The trials were corrected, from the start guess of the correction.
                self.assertIsNotNone(trials.rigid.guess)
                self.assertGreater(trials.rigid.guess.diagnostics()["counters"]["applications"], 0)
                # The track is engaged: evaluated, the flag would be True. Released it would be False,
                # and the preparation of the trials is the same: same signature, same graphs, no read.
                self.assertTrue(_eager_flag(interaction))
                graphs = [item[0] for item in trials.graphs.values()]
                self.assertEqual(len(graphs), 2)
                engaged = wp.clone(interaction._track_engaged)
                interaction._track_engaged.zero_()
                interaction._engaged_length = None
                try:
                    self.assertFalse(_eager_flag(interaction))
                    # An attempt of a step up to the preparation of its trials, ended as a rejected step ends.
                    attempt = coupler._open_attempt(scene.state_0, scene.control, DT, scene._external_forces)
                    try:
                        coupler._begin_attempt(attempt)
                        self.assertIs(coupler._device_trials(scene.state_0, scene.control, DT), trials)
                        trials.prepare(scene.state_0, scene.control, DT)
                        prepared = trials.signature
                    finally:
                        coupler._close_attempt(attempt, scene.state_0, False)
                finally:
                    wp.copy(interaction._track_engaged, engaged)
                    interaction._engaged_length = None
                self.assertEqual(prepared, signature)
                current = [item[0] for item in trials.graphs.values()]
                self.assertEqual(len(current), 2)
                for kept, graph in zip(graphs, current, strict=True):
                    self.assertIs(graph, kept)
                self.assertEqual(counts(), (0, 0, 0))
                self.assertIsNone(pc._active)
                # The next steps replay the two graphs.
                self._perturbed_step(scene)
                self._perturbed_step(scene)
                self.assertEqual((coupler.coupling_graph_captures, coupler.coupling_graph_fallbacks), (2, 0))
                self.assertEqual(counts(), (0, 0, 0))
            self.assertFalse(interaction.tunnel_alm._open)

    def test_host_trials_after_surface_metric_trials_evaluate_the_flag_once_as_begin_step_did(self):
        from newton_solver_suturebot import NeedleCouplingNotConverged
        from newton_solver_suturebot._src.solvers.coupled.host_trial_forces import TrialForceFallback

        with _surface_scene() as scene:
            coupler, interaction = scene.coupler, scene.interaction
            pc = coupler._shaft_preconditioner
            self._perturbed_step(scene)
            self._perturbed_step(scene)
            self.assertEqual((coupler.coupling_graph_captures, coupler.coupling_graph_fallbacks), (2, 0))
            self.assertFalse(coupler._captured_trials.evaluates_shaft_flag)
            with _engagement_reads(interaction) as counts:
                solve, accept = coupler._solve_host, interaction.accept
                seen, staged, accepted, raising = [], [], [], []

                def host(*args, **kwargs):
                    seen.append((args[5], pc._active, counts()))
                    if raising:
                        raise raising.pop()
                    return solve(*args, **kwargs)

                def refuse(*args, **kwargs):
                    staged.append(counts())
                    return False

                def counted_accept(state, body):
                    before = counts()
                    accept(state, body)
                    accepted.append(tuple(after - start for start, after in zip(before, counts(), strict=True)))

                # The converged device trial is refused where its endpoint is staged: the step falls
                # back to host-controlled trials, which are the first reader of the flag in this step.
                expected = _eager_flag(interaction)
                self.assertTrue(expected)
                with (
                    patch.object(interaction, "_stage_certified_trial", side_effect=refuse),
                    patch.object(coupler, "_solve_host", side_effect=host),
                    patch.object(interaction, "accept", side_effect=counted_accept),
                ):
                    self._perturbed_step(scene)
                    self.assertEqual(coupler.coupling_graph_fallbacks, 1)
                    self.assertEqual(coupler.coupling_graph_status, "exact GPU endpoint fallback")
                    # Nothing was read before the device trials ended.
                    self.assertEqual(staged[0], (0, 0, 0))
                    self.assertGreaterEqual(len(seen), 1)
                    # Every run of the host trials in the step got the value of the one evaluation.
                    for preconditioned, _, _ in seen:
                        self.assertIs(preconditioned, expected)
                    # The first of them found the flag held, after one query and one device read. No
                    # request: the step after a fallback is a device step again, which does not ask.
                    self.assertEqual(seen[0][1:], (expected, (1, 1, 0)))
                    # That read is the only one of the step: its acceptance queried and read nothing.
                    self.assertEqual((counts(), accepted), ((1, 1, 0), [(0, 0, 0)]))
                    self.assertFalse(interaction._engaged_length_requested)
                    self.assertIsNone(interaction._engaged_length)
                    # A second step in a row that falls back reads once again, and no more.
                    runs = len(seen)
                    self._perturbed_step(scene)
                    self.assertEqual(coupler.coupling_graph_fallbacks, 2)
                    self.assertEqual(seen[runs][1:], (expected, (2, 2, 0)))
                    self.assertEqual((counts(), accepted[1:]), ((2, 2, 0), [(0, 0, 0)]))
                    # A step whose captured host forces refuse it: its trials are restarted, which
                    # begins the step of the correction again, and the second run of the host trials
                    # finds the value of the step. Its corrections evaluate and read nothing.
                    runs = len(seen)
                    raising.append(TrialForceFallback("refused"))
                    self._perturbed_step(scene)
                    self.assertEqual(coupler.coupling_graph_fallbacks, 3)
                    self.assertEqual(seen[runs:], [(expected, expected, (3, 3, 0))] * 2)
                    self.assertEqual((counts(), accepted[2:]), ((3, 3, 0), [(0, 0, 0)]))
                    # A step whose host trials end in a rejection, and the same step again: nothing was
                    # accepted in between, so the second attempt is answered from the length that the
                    # interaction kept, by one more query and no read.
                    with (
                        patch.object(interaction, "check_step_resolution", side_effect=NeedleStepTooLarge("late")),
                        self.assertRaisesRegex(NeedleStepTooLarge, "late"),
                    ):
                        self._perturbed_step(scene)
                    self.assertEqual((coupler.coupling_graph_fallbacks, counts(), len(accepted)), (4, (4, 4, 0), 3))
                    self.assertIsNotNone(interaction._engaged_length)
                    self._perturbed_step(scene)
                    self.assertEqual(coupler.coupling_graph_fallbacks, 5)
                    self.assertEqual((counts(), accepted[3:]), ((5, 4, 0), [(0, 0, 0)]))
                    self.assertIsNone(interaction._engaged_length)
                # A device step after them evaluates and reads nothing.
                self._perturbed_step(scene)
                self.assertEqual((counts(), coupler.coupling_graph_fallbacks), ((5, 4, 0), 5))
                self.assertIsNone(pc._active)
                # A released track: the host trials get False, as begin_step evaluated it.
                engaged = wp.clone(interaction._track_engaged)
                interaction._track_engaged.zero_()
                interaction._engaged_length = None
                try:
                    self.assertFalse(_eager_flag(interaction))
                    with (
                        patch.object(coupler, "_device_trials", return_value=None),
                        patch.object(coupler, "_solve_host", return_value=False) as refused,
                        self.assertRaises(NeedleCouplingNotConverged),
                    ):
                        self._step(scene)
                    self.assertIs(refused.call_args.args[5], False)
                    # A step of this coupler without device trials is an exception as well: one read, no request.
                    self.assertEqual(counts(), (6, 5, 0))
                finally:
                    wp.copy(interaction._track_engaged, engaged)
                    interaction._engaged_length = None
            self.assertFalse(interaction.tunnel_alm._open)

    def test_graphs_captured_while_the_track_was_released_step_the_engaged_track_as_graphs_captured_for_it(self):
        # With the surface metric the flag of the shaft correction is no entry of the trial signature,
        # on the statement that the graph is the same for both of its values. Here the value differs
        # between the capture and the launches: one scene captures its two graphs at start-up while
        # its track is released (evaluated, the flag would be False), the other while the track is
        # engaged. Both then take the same steps from the same state with the engaged track. The
        # first launches the graphs of the released track in every step, captures nothing, corrects
        # its trials with the shaft correction and accepts the states of the second.
        with _surface_pair() as (plain, released):
            for scene in (plain, released):
                scene.state_1.assign(scene.state_0)
                self.assertTrue(_eager_flag(scene.interaction))
            interaction = released.interaction
            engaged = wp.clone(interaction._track_engaged)
            interaction._track_engaged.zero_()
            interaction._engaged_length = None
            try:
                self.assertFalse(_eager_flag(interaction))
                with _engagement_reads(interaction) as counts:
                    self.assertTrue(
                        released.coupler.capture_graphs((released.state_0, released.state_1), released.control, DT)
                    )
                    # Start-up did not evaluate the flag either.
                    self.assertEqual(counts(), (0, 0, 0))
            finally:
                wp.copy(interaction._track_engaged, engaged)
                interaction._engaged_length = None
            self.assertTrue(_eager_flag(interaction))
            self.assertTrue(plain.coupler.capture_graphs((plain.state_0, plain.state_1), plain.control, DT))
            kept = {}
            for scene in (plain, released):
                # The fixture read the flag when it was built, which asked the next acceptance for the
                # length, and start-up gave that request back. It is withdrawn: the steps are counted alone.
                self.assertTrue(scene.interaction._engaged_length_requested)
                scene.interaction._engaged_length_requested = False
                trials = scene.coupler._captured_trials
                self.assertFalse(trials.evaluates_shaft_flag)
                self.assertEqual((trials.captures, len(trials.graphs)), (2, 2))
                signature = trials.signature
                self.assertIsNone(signature[signature.index(scene.coupler._shaft_preconditioner.storage_key()) + 1])
                kept[id(scene)] = (signature, [entry[0] for entry in trials.graphs.values()])
            with _engagement_reads(interaction) as counts:
                for step in range(4):
                    for scene in (plain, released):
                        # A carried load that the first trial does not balance: the trials need the correction.
                        load = scene.coupler._load.numpy().copy()
                        load[-6] += 1e-4
                        scene.coupler._load.assign(load)
                        self._step(scene)
                        coupler, trials = scene.coupler, scene.coupler._captured_trials
                        self.assertEqual(coupler.coupling_graph_fallbacks, 0)
                        # The graphs of start-up are the ones launched: nothing is captured again.
                        signature, graphs = kept[id(scene)]
                        self.assertEqual(
                            (coupler.coupling_graph_captures, coupler.coupling_graph_launches), (2, step + 1)
                        )
                        self.assertEqual(trials.signature, signature)
                        current = [entry[0] for entry in trials.graphs.values()]
                        self.assertEqual(len(current), 2)
                        for graph, captured in zip(current, graphs, strict=True):
                            self.assertIs(graph, captured)
                        self.assertIsNone(coupler._shaft_preconditioner._active)
                    # The same number of trials, and the accepted states agree as two runs of one scene do.
                    self.assertEqual(released.coupler.iterations, plain.coupler.iterations, step)
                    self.assertEqual(history(released.interaction), history(plain.interaction))
                    for name in FIELDS:
                        velocity = name.endswith("qd")
                        np.testing.assert_allclose(
                            getattr(released.state_1, name).numpy(),
                            getattr(plain.state_1, name).numpy(),
                            rtol=0 if velocity else 3e-5,
                            atol=2e-5 if velocity else 2e-7,
                            err_msg=f"step {step}: {name}",
                        )
                    for scene in (plain, released):
                        scene.state_0, scene.state_1 = scene.state_1, scene.state_0
                self.assertEqual(counts(), (0, 0, 0))
            # The trials of both scenes were corrected, from the start guess of the shaft correction.
            applications = []
            for scene in (plain, released):
                guess = scene.coupler._captured_trials.rigid.guess
                self.assertIsNotNone(guess)
                applications.append(guess.diagnostics()["counters"]["applications"])
            self.assertGreater(applications[1], 0)
            self.assertEqual(applications[1], applications[0])
            self.assertFalse(interaction.tunnel_alm._open)

    def test_trials_without_a_surface_metric_keep_the_flag_as_an_entry_of_their_signature(self):
        # Without the surface metric an inactive correction is not applied and the loop relaxes
        # the residual itself: the graph follows the flag, which stays an entry and is evaluated
        # before every launch (answered from the read of the acceptance before it).
        with _pair() as (_, captured):
            coupler, interaction = captured.coupler, captured.interaction
            pc = coupler._shaft_preconditioner
            self.assertIsNone(coupler._surface_preconditioner)
            self._step(captured)
            captured.state_0, captured.state_1 = captured.state_1, captured.state_0
            trials = coupler._captured_trials
            self.assertFalse(trials.rigid.surface_preconditioned)
            signature = trials.signature
            entry = signature.index(pc.storage_key()) + 1
            self.assertIs(signature[entry], True)
            with _engagement_reads(interaction) as counts:
                self._step(captured)
                # One query before the launch, answered without a device read; the acceptance read once.
                self.assertEqual(counts(), (1, 1, 1))
                self.assertIs(pc._active, True)
            self.assertEqual(coupler.coupling_graph_fallbacks, 0)

    def test_trials_without_a_surface_metric_and_a_response_basis_capture_the_graph_of_each_flag_value(self):
        # One graph with a device branch on the flag is the native free needle's alone. On MuJoCo
        # Warp without the surface metric and without a response-basis predictor the rigid
        # adapter applies the correction itself, and the graph holds the correction of one value:
        # the flag is an entry of the signature, a flip drops the graphs of both buffers and the
        # next two steps capture those of the other value, and nothing writes the device flag
        # of the branch.
        with patch.object(GraspScene, "response_basis_guesses", False), _pair() as (_, captured):
            coupler, interaction = captured.coupler, captured.interaction
            pc = coupler._shaft_preconditioner
            self.assertIsNone(coupler._surface_preconditioner)
            self.assertIsNone(getattr(captured.solver, "_affine_predictor", None))

            def step():
                self._step(captured)
                captured.state_0, captured.state_1 = captured.state_1, captured.state_0
                signature = coupler._captured_trials.signature
                return signature[signature.index(pc.storage_key()) + 1], coupler.coupling_graph_captures

            steps = [step(), step()]
            trials = coupler._captured_trials
            engaged = [entry[0] for entry in trials.graphs.values()]
            # The track reported as released: the flag is False in these two steps.
            with patch.object(type(interaction), "engaged_track_length", new_callable=PropertyMock, return_value=0.0):
                steps += [step(), step()]
            released = [entry[0] for entry in trials.graphs.values()]
            steps += [step(), step()]
            entries, captures = zip(*steps, strict=True)
            for entry, expected in zip(entries, (True, True, False, False, True, True), strict=True):
                self.assertIs(entry, expected)
            self.assertEqual(captures, (1, 2, 3, 4, 5, 6))
            self.assertEqual(coupler.coupling_graph_launches, 6)
            self.assertEqual(coupler.coupling_graph_fallbacks, 0)
            self.assertIs(coupler._captured_trials, trials)
            self.assertIs(type(trials), CapturedArticulatedTrials)
            self.assertTrue(hasattr(trials.rigid, "apply_preconditioner"))
            self.assertFalse(trials.rigid.surface_preconditioned)
            self.assertTrue(trials.evaluates_shaft_flag)
            self.assertFalse(trials.branches_on_shaft_flag)
            current = [entry[0] for entry in trials.graphs.values()]
            for graphs in (engaged, released, current):
                self.assertEqual(len(graphs), 2)
            for graph in released + current:
                self.assertFalse(any(graph is kept for kept in engaged))
            for graph in current:
                self.assertFalse(any(graph is kept for kept in released))
            np.testing.assert_array_equal(trials.shaft_active.numpy(), [0])
            self.assertIs(trials._shaft_active, False)
            for name in FIELDS:
                self.assertTrue(np.isfinite(getattr(captured.state_0, name).numpy()).all(), name)

    def test_replacing_actuator_target_storage_invalidates_outer_graph_binding(self):
        with _pair() as (_, captured):
            self._step(captured)
            before = captured.coupler.coupling_graph_captures
            captured.state_0, captured.state_1 = captured.state_1, captured.state_0
            replacement = wp.clone(captured.control.joint_target_q)
            captured.control.joint_target_q = replacement
            self._step(captured)
            self.assertGreater(captured.coupler.coupling_graph_captures, before)
            self.assertEqual(captured.coupler.coupling_graph_fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
