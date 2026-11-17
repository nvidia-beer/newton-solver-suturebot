# SPDX-License-Identifier: Apache-2.0
"""Grasp/FEM transactions and physical-thread lifecycle on a small real mesh."""

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from test_needle_insertion_example import _REFERENCE, _selection

from newton_solver_suturebot import NeedleStepTooLarge
from newton_solver_suturebot._src.solvers.coupled.host_trial_prepare import _arrays
from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco_warp import INTEGRATION_FIELDS
from newton_solver_suturebot._src.utils.capture import census, loaded_modules
from newton_solver_suturebot.examples.grasp._grasp_scene import GraspScene
from newton_solver_suturebot.examples.grasp.parallel import (
    LocalSutureScene,
    SutureSceneConfig,
    _apply_action,
    _geometry,
    _make_scene,
    _mark_startup,
    _observe,
    _reset_scene,
    _snapshot,
    _step,
)
from newton_solver_suturebot.examples.phantom import example_fem_phantom_regions as phantom


@contextmanager
def _scene(**settings):
    with (
        TemporaryDirectory() as directory,
        patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection),
    ):
        config = SutureSceneConfig(
            device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"),
            asset=str(Path(directory) / "pad.usd"),
            entry_xy=(0, 0.005),
            exit_xy=(0, -0.005),
            control_mode="physical_grasp",
            **settings,
        )
        scene = _make_scene(config, 0)
        try:
            yield scene, config
        finally:
            scene.viewer.close()


def _native(scene):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(scene.rigid_solver.mj_model, specification))
    mujoco.mj_getState(scene.rigid_solver.mj_model, scene.rigid_solver.mj_data, values, specification)
    return values


def _bits(array):
    return str(array.dtype), tuple(array.shape), array.numpy().tobytes()


def _startup_record(scene):
    """What the start-up call of a scene has to leave as it found it: device arrays as bits, and host values."""
    solver, coupler, thread = scene.solver, scene.coupler, scene.worker_thread
    arrays = {f"state_0.{name}": array for name, array in _arrays(scene.state_0)}
    arrays.update({f"control.{name}": array for name, array in _arrays(scene.control)})
    arrays.update({f"integration.{name}": getattr(scene.rigid_solver.mjw_data, name) for name in INTEGRATION_FIELDS})
    # Every issued rigid interval writes its microstep here (MuJoCo Warp's opt.timestep on the device).
    arrays["device time step"] = scene.rigid_solver.mjw_model.opt.timestep
    arrays["FEM velocity increment"] = solver.velocity_increment
    arrays["coupler._load"] = coupler._load
    arrays["held start uses"] = solver._held_start.uses
    multipliers = scene.interaction.tunnel_alm
    arrays.update({f"multipliers.{name}": getattr(multipliers, name) for name, _ in multipliers._ARRAYS})
    arrays.update({f"thread.{index}": array for index, array in enumerate((*thread.state, thread.solver.iterations))})
    arrays.update(
        {
            f"IK guess {index}": ik.guess
            for index, ik in enumerate(scene.pose_ik)
            if isinstance(getattr(ik, "guess", None), wp.array)
        }
    )
    host = {
        "elapsed": scene.elapsed,
        "accepted steps": (scene.accepted_physics_steps, thread.accepted_steps, thread.physical_time_s),
        "refinements": scene.step_refinements,
        "events": (list(scene.interaction_events), tuple(getattr(scene.interaction, "events", ()))),
        "no accepted tip": scene.interaction._previous_tip is None,
        "history": (scene.interaction.track_count, scene.interaction.opening_count, scene.interaction.status),
        "engaged length requested": scene.interaction._engaged_length_requested,
        "multipliers": (
            multipliers._open,
            multipliers.dual_updates,
            multipliers.total_dual_updates,
            multipliers.normal_error,
            multipliers.dual_residual,
        ),
        "native timestep": scene.rigid_solver.mj_model.opt.timestep,
        "native shadow token": scene.rigid_solver._shadow_token,
        "tetrahedron query committed": getattr(scene.interaction._device_queries, "has_previous", False),
        "step time": coupler.step_time.value,
        "thread contact notes": (thread.rigid_contact._body_q, thread.rigid_contact._dt),
        "FEM tolerance": solver.linear_solver_tolerance,
        "arm targets": [np.asarray(target).tobytes() for target in scene.arm_targets],
    }
    return {name: _bits(array) for name, array in arrays.items()}, host, scene.report()


@contextmanager
def _recorded_startup():
    """Record every scene around its start-up call; yields the list of (before, after, returned)."""
    records = []
    call = GraspScene.capture_graphs

    def recorded(scene):
        before = _startup_record(scene)
        returned = call(scene)
        records.append((before, _startup_record(scene), returned))
        return returned

    with patch.object(GraspScene, "capture_graphs", recorded):
        yield records


def _tangent(scene):
    """The assembled tangent blocks of the FEM solver; the storage beyond them is not written."""
    matrix = scene.solver.A_bsr
    return matrix.values.numpy()[: matrix.nnz_sync()]


def _flat(value, prefix=""):
    """Leaves of a report as {dotted name: value}."""
    if isinstance(value, dict):
        return {name: leaf for key, item in value.items() for name, leaf in _flat(item, f"{prefix}{key}.").items()}
    return {prefix[:-1]: value}


# Report entries that may differ across a start-up call: capture counters, graph status texts, the
# census, the number of scratch buffers of the rigid step, which exists after its first interval, the
# packet rows of the host Anderson workspace, which exists after the call, and the counters of the
# correction inside the trial graph, which exist with the owner of that graph.
def _startup_may_change(name):
    leaf = name.rsplit(".", 1)[-1]
    return (
        leaf.endswith("captures")
        or leaf.endswith("graph_status")
        or leaf in ("gpu_scratch_buffers", "device_anderson_packet_capacity")
        or name == "gpu_coupling_anderson_counters"
        or name.startswith("capture_census.")
        or leaf in ("module_loads_after_startup", "render_module_loads_after_startup")
    )


# Every variant of a host force evaluation: (surface recorder bound, residual of the trial before, new candidate).
_FORCE_VARIANTS = sorted(
    (surface, previous, fresh) for surface in (False, True) for previous in (False, True) for fresh in (False, True)
)
# Source files whose graphs the start-up call captures for the host-controlled trials and their correction.
_HOST_PATH_FILES = (
    "host_trial_forces.py",
    "shaft_preconditioner.py",
    "contact_preparation.py",
    "surface_preconditioner.py",
)


def _host_path_sites():
    """Captures of this process, by site, in the files of the host-path graphs (Krylov captures under them included)."""
    return {site: count for site, count in census.sites.items() if site.rsplit(":", 1)[0].endswith(_HOST_PATH_FILES)}


def _host_path_modules():
    """Loaded kernel module builds of the host force path, the two corrections and the host Anderson workspace."""
    names = (
        "host_trial_forces",
        "needle_trial_forces",
        "interface.insertion",
        "needle_queries",
        "surface_contact_forces",
        "surface_facet_scratch",
        "surface_preconditioner",
        "shaft_preconditioner",
        "surface_anderson_device",
        "certified_scope",
    )
    return {key: digest for key, digest in loaded_modules().items() if key[0].endswith(names)}


# What a scene whose start-up call ran reports as zero when nothing was captured or loaded after it.
_LATE_NUMBERS = ("captures_after_startup", "solver_captures_after_startup", "module_loads_after_startup")


@contextmanager
def _counted_captures():
    """Count every CUDA capture that Warp begins, whoever asks for it; yields the list of their callers.

    The package counts the captures of its own scopes and the Krylov solves issued outside a
    capture. This counts at Warp's entry, as the external census tool does, so it also sees a
    stock ``wp.ScopedCapture`` that no package scope encloses.
    """
    begun, begin = [], wp.capture_begin
    warp_source = os.path.dirname(wp.__file__) + os.sep

    def counting(device=None, stream=None, *arguments, **options):
        target = stream.device if stream is not None else wp.get_device(device)
        if target.is_cuda:
            # The caller is the first frame outside Warp: the code that opened the capture scope,
            # or that issued the Krylov solve for which Warp captures a graph of its own.
            frame = sys._getframe(1)
            while frame.f_back is not None and frame.f_code.co_filename.startswith(warp_source):
                frame = frame.f_back
            begun.append(f"{frame.f_code.co_filename}:{frame.f_lineno}")
        return begin(device, stream, *arguments, **options)

    with patch.object(wp, "capture_begin", counting):
        yield begun


def _late(diagnostics):
    """The three numbers of a report or of an observation's diagnostics, by name."""
    return {name: diagnostics[name] for name in _LATE_NUMBERS}


_NOTHING_LATE = dict.fromkeys(_LATE_NUMBERS, 0)


class GraspSceneTests(unittest.TestCase):
    def test_observation_reads_each_rigid_array_once_and_preserves_diagnostics(self):
        with _scene() as (scene, _):
            scene.step()
            expected = scene.observe_grasp()
            arrays = {id(getattr(scene.state_0, name)): name for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
            reads = dict.fromkeys(arrays.values(), 0)
            original = wp.array.numpy

            def read(array):
                name = arrays.get(id(array))
                if name is not None:
                    reads[name] += 1
                return original(array)

            with patch.object(wp.array, "numpy", new=read):
                observed = _observe(scene, 0)
            self.assertEqual(reads, dict.fromkeys(reads, 1))
            for name, expected_value in expected.items():
                if isinstance(expected_value, np.ndarray):
                    np.testing.assert_array_equal(observed[name], expected_value)
                else:
                    self.assertEqual(observed[name], expected_value)
            np.testing.assert_array_equal(observed["needle_pose"], observed["rigid_body_poses"][scene.needle_body])
            np.testing.assert_array_equal(observed["tip_position"], scene.actual_tip())
            # Readbacks are scoped to this observation, never retained across
            # physical advancement or a swap of accepted device-state buffers.
            scene.step()
            current = _observe(scene, 0)
            np.testing.assert_array_equal(current["rigid_body_poses"], scene.state_0.body_q.numpy())
            np.testing.assert_array_equal(
                current["arm_joint_positions"],
                [scene.state_0.joint_q.numpy()[list(tool.arm_q_indices)] for tool in scene.graspers],
            )

    def test_thread_tracks_are_the_scenes_default_and_the_switch_gives_the_worker_of_before(self):
        with _scene() as (scene, config):
            worker = scene.worker_thread
            self.assertTrue(scene.args.thread_tracks)
            self.assertTrue(worker.thread_tracks)
            # The reserve: per bite one eyelet per spacing of needle and two per planned passage; two bites.
            passages = scene.plan.expected_passages
            length = float(np.linalg.norm(np.diff(scene.geometry.centerline, axis=0), axis=1).sum())
            reserve = (2 * (int(length // worker.eyelet_spacing) + 2 * passages), 2 * passages)
            self.assertEqual(worker.eyelet_reserve, reserve)
            self.assertEqual((worker.solver.max_eyelets, len(worker.tissue._group_size)), reserve)
            self.assertEqual(worker.solver.eyelet_stiffness, worker.contact_stiffness)
            # The follower reads this scene's interaction, and the coupling follows the needle's tail.
            self.assertIs(worker.track.scene, scene)
            self.assertIs(worker.track.coupling, worker.tissue)
            body, _point, _outward, along, lateral = worker.tissue._tail
            self.assertEqual((body, along), (scene.needle_body, np.float32(worker.eyelet_spacing)))
            self.assertAlmostEqual(lateral, 2.0 * scene.geometry.thickness, delta=1e-9)
            # The interaction's tetrahedra are the coupling's, in the same order: a track row names both.
            np.testing.assert_array_equal(scene.interaction.tetrahedra, worker.tissue._tetrahedra.numpy())
            geometry = _geometry(scene, config)
            self.assertIs(geometry["thread_tracks"], True)
            self.assertEqual(geometry["thread_element_length"], config.thread_length_m / config.thread_elements)
            self.assertIn("needle's actual track", scene.report()["thread_coupling"])
            scene.step()
            report = scene.observe_grasp()["thread_diagnostics"]
            self.assertEqual(
                [report[name] for name in ("eyelets", "eyelets_threaded", "eyelets_refused", "eyelet_reserve")],
                [0, 0, 0, list(reserve)],
            )
        with _scene(example_arguments=("--no-thread-tracks",)) as (scene, config):
            worker = scene.worker_thread
            self.assertFalse(scene.args.thread_tracks)
            self.assertEqual((worker.thread_tracks, worker.eyelet_reserve, worker.track), (False, (0, 1), None))
            self.assertEqual((worker.solver.max_eyelets, len(worker.tissue._group_size)), (0, 1))
            self.assertIsNone(worker.tissue._tail)
            geometry = _geometry(scene, config)
            self.assertIs(geometry["thread_tracks"], False)
            self.assertIn("no scheduled eyelets", scene.report()["thread_coupling"])
            scene.step()
            self.assertEqual(scene.observe_grasp()["thread_diagnostics"]["eyelets"], 0)

    def test_incremental_actions_do_not_reposition_independent_needle_or_thread(self):
        with _scene() as (scene, config):
            initial = _observe(scene, 0)
            rigid = scene.state_0.body_q.numpy().copy()
            thread = scene.worker_thread.positions().copy()
            geometry = _geometry(scene, config)
            self.assertEqual(config.action_dimension, 7)
            self.assertEqual(scene.model.body_count, 17)
            self.assertEqual(scene.rigid_solver.mj_model.neq, 0)
            self.assertNotEqual(geometry["needle_body_index"], geometry["gripper_body_index"])
            self.assertEqual(scene.coupler.rigid_transfer_mode, "general")
            self.assertEqual(len(geometry["grippers"]), 2)
            report = initial["diagnostics"]
            self.assertEqual(report["control_mode"], "physical_grasp")
            self.assertTrue(report["independent_free_needle"])
            self.assertFalse(report["needle_pose_drive"])
            self.assertIn("fixed trocar", report["shaft_constraint"])
            self.assertIn("joint position actuators", report["pose_control"])
            self.assertNotIn("drive_stiffness_N_per_m", report)
            self.assertEqual(initial["arm_joint_positions"].shape, (2, 6))
            for item in initial["rcm_diagnostics"]:
                self.assertLess(item["shaft_to_trocar_distance_m"], 1e-6)
                self.assertTrue(item["trocar_on_shaft"])
            self.assertEqual(report["native_rigid_substeps"], 4)
            self.assertEqual(report["native_contact_cone"], "elliptic")
            self.assertEqual(scene.rigid_solver.mj_model.opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
            self.assertEqual(report["native_solver_iterations"], 200)
            self.assertEqual(report["native_solver_line_search_iterations"], 50)
            self.assertAlmostEqual(report["native_solver_tolerance"], 1e-12, delta=1e-18)
            self.assertAlmostEqual(report["native_rigid_dt_s"], config.frame_dt / (config.substeps * 4))
            _apply_action(scene, config, [0.1, 0, 0, 0, 0, 0, 1])
            self.assertAlmostEqual(scene.jaw_opening_target, config.jaw_step_rad)
            target = scene.cartesian_target.copy()
            _apply_action(scene, config, np.zeros(7))
            self.assertAlmostEqual(scene.jaw_opening_target, config.jaw_step_rad)
            np.testing.assert_array_equal(scene.cartesian_target, target)
            np.testing.assert_array_equal(scene.state_0.body_q.numpy(), rigid)
            np.testing.assert_array_equal(scene.worker_thread.positions(), thread)
            self.assertEqual(scene.worker_thread.accepted_steps, 0)
            with self.assertRaises(ValueError):
                _apply_action(scene, config, np.zeros(6))
            scene.step()
            self.assertEqual(scene.worker_thread.accepted_steps, scene.accepted_physics_steps)
            self.assertAlmostEqual(scene.worker_thread.physical_time_s, scene.elapsed)
            self.assertGreater(scene.elapsed, 0)
            self.assertTrue(np.isfinite(scene.worker_thread.positions()).all())
            self.assertTrue(np.isfinite(scene.state_0.body_q.numpy()).all())
            scene._check_tissue_integrity()
            current = _observe(scene, 0)
            np.testing.assert_array_equal(current["trocar_poses"], initial["trocar_poses"])
            for item in current["rcm_diagnostics"]:
                self.assertLess(item["shaft_to_trocar_distance_m"], 1e-6)
                self.assertTrue(item["trocar_on_shaft"])
            _reset_scene(scene, config, rebuild=True)
            reset = _observe(scene, 0)
            self.assertEqual(reset["reset_count"], initial["reset_count"] + 1)
            self.assertEqual(reset["thread_diagnostics"]["accepted_steps"], 0)
            self.assertEqual(reset["thread_diagnostics"]["physical_time_s"], 0)
            np.testing.assert_array_equal(reset["rigid_body_poses"], rigid)
            np.testing.assert_array_equal(reset["thread_positions"], thread)

    def test_rejected_multi_body_trial_restores_native_and_output_without_advancing_thread(
        self,
    ):
        with _scene() as (scene, _):
            scene.state_0.clear_forces()
            scene._prepare_step_loads(scene.cartesian_target, np.zeros(6), scene.sim_dt)
            scene.solver.refresh_tangent(scene.state_0.particle_q, scene.sim_dt)
            native = _native(scene)
            saved = {
                name: getattr(scene.state_1, name).numpy().copy()
                for name in (
                    "joint_q",
                    "joint_qd",
                    "body_q",
                    "body_qd",
                    "particle_q",
                    "particle_qd",
                )
            }
            forces = scene.state_0.body_f.numpy().copy()
            thread = [array.numpy().copy() for array in scene.worker_thread.state]
            with (
                patch.object(
                    scene.coupler._tet_step_guard,
                    "check",
                    side_effect=NeedleStepTooLarge("test rejection"),
                ),
                self.assertRaises(NeedleStepTooLarge),
            ):
                scene.coupler.step(
                    scene.state_0,
                    scene.state_1,
                    scene.control,
                    scene.sim_dt,
                    external_forces=scene._external_forces,
                )
            np.testing.assert_array_equal(_native(scene), native)
            for name, expected in saved.items():
                np.testing.assert_array_equal(getattr(scene.state_1, name).numpy(), expected)
            np.testing.assert_array_equal(scene.state_0.body_f.numpy(), forces)
            for expected, array in zip(thread, scene.worker_thread.state, strict=True):
                np.testing.assert_array_equal(array.numpy(), expected)
            self.assertEqual(scene.worker_thread.accepted_steps, 0)
            self.assertEqual(scene.elapsed, 0)

    def test_adaptive_retry_advances_thread_only_for_accepted_intervals(self):
        with _scene() as (scene, config):
            check = scene.coupler._tet_step_guard.check
            attempts = 0

            def once(state_in, candidate):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise NeedleStepTooLarge("one unresolved candidate")
                return check(state_in, candidate)

            with patch.object(scene.coupler._tet_step_guard, "check", side_effect=once):
                scene.step()
            self.assertEqual(scene.step_refinements, 1)
            self.assertEqual(scene.coupler.tet_step_rejections, 1)
            self.assertEqual(attempts, scene.accepted_physics_steps + 1)
            self.assertEqual(scene.worker_thread.accepted_steps, scene.accepted_physics_steps)
            self.assertAlmostEqual(scene.elapsed, config.frame_dt)
            self.assertAlmostEqual(scene.worker_thread.physical_time_s, config.frame_dt)
            self.assertTrue(scene.running, scene.failure_reason)

    def test_error_while_an_attempt_opens_is_not_closed_and_one_while_it_begins_is(self):
        # The close of an attempt restores what its open phase saved. An error of the open phase
        # leaves the saved values of the attempt BEFORE it, so it must not be closed; an error of
        # the begin phase follows a complete open phase and must be.
        with _scene() as (scene, _):
            scene.step()
            self.assertTrue(scene.running, scene.failure_reason)
            coupler, multipliers = scene.coupler, scene.interaction.tunnel_alm
            scene.state_0.clear_forces()
            scene._prepare_step_loads(scene.cartesian_target, np.zeros(6), scene.sim_dt)
            native = _native(scene)
            kept = {
                "increment": scene.solver.velocity_increment.numpy().copy(),
                "particle_f": scene.state_0.particle_f.numpy().copy(),
                "body_f": scene.state_0.body_f.numpy().copy(),
            }

            def check_kept():
                np.testing.assert_array_equal(_native(scene), native)
                np.testing.assert_array_equal(scene.solver.velocity_increment.numpy(), kept["increment"])
                np.testing.assert_array_equal(scene.state_0.particle_f.numpy(), kept["particle_f"])
                np.testing.assert_array_equal(scene.state_0.body_f.numpy(), kept["body_f"])

            closed = []
            close = coupler._close_attempt

            def closing(attempt, state_in, committed):
                closed.append(committed)
                return close(attempt, state_in, committed)

            with patch.object(coupler, "_close_attempt", side_effect=closing):
                # Open phase: nodal forces of the wrong size are refused before the rigid state is saved.
                wrong = wp.zeros(scene.model.particle_count + 1, dtype=wp.vec3, device=scene.device)
                with self.assertRaises(RuntimeError):
                    coupler.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt, external_forces=wrong)
                self.assertEqual(closed, [])
                # A close would have put back the rigid state and the FEM increment of the accepted
                # step before this one.
                check_kept()
                self.assertFalse(multipliers._open)
                # Begin phase: its last call raises, after the native input began and the multipliers opened.
                inputs = []
                begin = scene.rigid_solver.begin_repeated_input

                def beginning(*arguments):
                    inputs.append(begin(*arguments))
                    return inputs[-1]

                with (
                    patch.object(scene.rigid_solver, "begin_repeated_input", side_effect=beginning),
                    patch.object(coupler._shaft_preconditioner, "begin_step", side_effect=RuntimeError("begin failed")),
                    self.assertRaisesRegex(RuntimeError, "begin failed"),
                ):
                    coupler.step(
                        scene.state_0,
                        scene.state_1,
                        scene.control,
                        scene.sim_dt,
                        external_forces=scene._external_forces,
                    )
                self.assertEqual(closed, [False])
            # Closed like a rejected step: the multipliers are discarded, the native input is ended,
            # and what the open phase saved is back.
            self.assertFalse(multipliers._open)
            self.assertEqual(len(inputs), 1)
            if inputs[0] is not None:
                self.assertFalse(inputs[0]._repeated_active)
            check_kept()
            self.assertEqual(scene.worker_thread.accepted_steps, scene.accepted_physics_steps)
            # The next frame runs: nothing was left open.
            steps = scene.accepted_physics_steps
            scene.step()
            self.assertTrue(scene.running, scene.failure_reason)
            self.assertGreater(scene.accepted_physics_steps, steps)

    def test_scene_on_the_native_backend_has_no_start_up_call(self):
        with _scene() as (scene, config):
            # The reset asked the scene; it declined, and nothing is checked after start-up.
            self.assertFalse(scene._startup_ready)
            self.assertIsNone(scene._startup_mark)
            self.assertIsNone(scene._startup_tangent)
            self.assertEqual(scene.report()["capture_census"]["status"], "native rigid backend")
            scene._startup_status = "start-up not marked"
            with (
                patch.object(scene.solver, "refresh_tangent", side_effect=AssertionError("hidden tangent refresh")),
                patch.object(scene.worker_thread, "capture_graphs", side_effect=AssertionError("hidden thread step")),
                patch.object(scene.coupler, "capture_graphs", side_effect=AssertionError("hidden attempt")),
            ):
                self.assertFalse(scene.capture_graphs())
            self.assertEqual(scene._startup_status, "native rigid backend")
            # The entries of the FEM step and of the trial graph issue nothing where nothing is captured.
            with patch.object(scene.solver, "step", side_effect=AssertionError("hidden FEM step")):
                self.assertEqual(scene.coupler.warm_fem(scene.state_0, scene.control, scene.sim_dt), 0)
            attempt = scene.coupler._open_attempt(scene.state_0, scene.control, scene.sim_dt, None)
            try:
                self.assertFalse(scene.coupler.capture_trial_graph(attempt))
            finally:
                scene.coupler._close_attempt(attempt, scene.state_0, False)
            self.assertIsNone(scene.coupler._captured_trials)
            # The coupler's own entry declines for a rigid solver without one, and opens no attempt.
            with patch.object(scene.coupler, "_open_attempt", side_effect=AssertionError("hidden attempt")):
                self.assertFalse(
                    scene.coupler.capture_graphs((scene.state_0, scene.state_1), scene.control, scene.sim_dt)
                )
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    scene.coupler.capture_graphs((scene.state_0, scene.state_1), scene.control, 0.0)
            self.assertIsNone(scene.coupler._host_trial_preparation)
            self.assertEqual(
                (scene.elapsed, scene.accepted_physics_steps, scene.worker_thread.accepted_steps), (0, 0, 0)
            )
            _reset_scene(scene, config, rebuild=True)
            self.assertFalse(scene._startup_ready)
            self.assertEqual(scene._startup_status, "native rigid backend")

    def test_tangent_built_ahead_serves_the_first_frame_once_and_only_while_it_is_current(self):
        def frame(scene):
            refresh, calls = scene.solver.refresh_tangent, []

            def refreshing(particle_q, dt):
                calls.append(dt)
                return refresh(particle_q, dt)

            with patch.object(scene.solver, "refresh_tangent", side_effect=refreshing):
                scene.step()
            self.assertTrue(scene.running, scene.failure_reason)
            self.assertEqual(scene.step_refinements, 0)
            return calls

        def ahead(scene):
            # What a start-up call does for the first frame.
            wp.copy(scene._frame_particle_q, scene.state_0.particle_q)
            scene.solver.refresh_tangent(scene._frame_particle_q, scene.sim_dt)
            return (scene.solver._tangent_version, scene.sim_dt, scene.accepted_physics_steps)

        names = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
        with _scene() as (plain, _), _scene() as (prepared, config):
            self.assertEqual(frame(plain), [plain.sim_dt])
            prepared._startup_tangent = ahead(prepared)
            # The frame uses the tangent built ahead in place of its own refresh, once.
            self.assertEqual(frame(prepared), [])
            self.assertIsNone(prepared._startup_tangent)
            if not prepared.device.is_cuda:
                # Same positions, time step and code path: the frame is the same frame, bit for bit.
                np.testing.assert_array_equal(_tangent(prepared), _tangent(plain))
                for name in names:
                    np.testing.assert_array_equal(
                        getattr(prepared.state_0, name).numpy(), getattr(plain.state_0, name).numpy(), err_msg=name
                    )
            self.assertEqual(frame(prepared), [prepared.sim_dt])
            # A token is not used when a step was accepted, the tangent was written or the time step
            # differs since it was taken; it is dropped all the same.
            for stale in (
                lambda token: (token[0], token[1], token[2] - 1),
                lambda token: (token[0] - 1, token[1], token[2]),
                lambda token: (token[0], token[1] / 2, token[2]),
            ):
                token = (prepared.solver._tangent_version, prepared.sim_dt, prepared.accepted_physics_steps)
                prepared._startup_tangent = stale(token)
                self.assertEqual(frame(prepared), [prepared.sim_dt])
                self.assertIsNone(prepared._startup_tangent)
            # A rebuilt scene drops the token of the scene before it.
            prepared._startup_tangent = ahead(prepared)
            _reset_scene(prepared, config, rebuild=True)
            self.assertIsNone(prepared._startup_tangent)
            self.assertEqual(frame(prepared), [prepared.sim_dt])


class GraspSceneStartupTests(unittest.TestCase):
    """The start-up call of the scene on the MuJoCo Warp rigid backend (CUDA with conditional graphs)."""

    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The start-up call captures CUDA graphs")

    def check_record(self, scene, record):
        (arrays, host, report), (arrays_after, host_after, report_after), returned = record
        self.assertTrue(returned)
        self.assertEqual(sorted(arrays_after), sorted(arrays))
        for name, bits in arrays.items():
            self.assertEqual(arrays_after[name], bits, name)
        self.assertEqual(host_after, host)
        self.assertEqual(host["elapsed"], 0.0)
        self.assertEqual(host["accepted steps"], (0, 0, 0.0))
        self.assertTrue(host["no accepted tip"])
        before, after = _flat(report), _flat(report_after)
        self.assertEqual(sorted(after), sorted(before))
        changed = sorted(name for name in before if before[name] != after[name])
        self.assertEqual([name for name in changed if not _startup_may_change(name)], [])
        # Work counters, among them those of the trial loops that start-up does not run.
        for name in (
            "coupling_divergence_stops",
            "coupling_graph_launches",
            "coupling_graph_fallbacks",
            "fem_graph_launches",
            "trial_prepare_graph_launches",
            "held_fem_start_preparations",
            "certified_linear_attempts",
            "tunnel_alm_dual_updates",
            "maximum_coupling_iterations",
            "last_coupling_attempt_iterations",
            "gpu_rigid_diagnostics.gpu_graph_launches",
            "response_basis_builds",
            "response_reference_builds",
        ):
            self.assertEqual((name, after[name]), (name, 0))
        self.assertEqual(scene.coupler.coupling_unresolved_trials, 0)
        # The trial launched at start-up made no correction: the counters of its mixing are zeros.
        self.assertIsNone(before["gpu_coupling_anderson_counters"])
        self.assertFalse(any(after["gpu_coupling_anderson_counters"] or ()))
        # What the call captured, each launched once and counted by its owner as a capture alone.
        self.assertEqual(after["trial_prepare_graph_captures"], 2)
        self.assertEqual(after["gpu_rigid_diagnostics.gpu_graph_captures"], 2)
        # The trial graph of each buffer, and the FEM step of a host trial at every inner tolerance.
        self.assertEqual(after["coupling_graph_captures"], 2)
        self.assertEqual(after["fem_graph_captures"], 2 * len(scene.solver.tolerance_levels()))
        # The host force graphs and the graphs of the host-controlled correction.
        for name, captures in (
            ("host_force_graph_captures", 8),
            ("surface_pc_graph_captures", 1),
            ("shaft_pc_graph_captures", 2),
            ("contact_preparation_captures", 1),
        ):
            self.assertEqual((name, after[name]), (name, captures))
        for name in (
            "host_force_graph_launches",
            "host_force_graph_fallbacks",
            "host_force_primal_reuses",
            "surface_pc_graph_launches",
            "surface_pc_native_mass_fallbacks",
            "surface_pc_applications",
            "surface_pc_selection_changes",
            "contact_preparation_launches",
            "contact_preparation_eager",
            "woodbury_packet_preparation_launches",
            "woodbury_packet_preparation_eager",
            "device_anderson_transport_calls",
        ):
            self.assertEqual((name, after[name]), (name, 0))
        # The workspace of the host Anderson mixing exists now, with the packet of its default rows.
        rows = min(128, scene.coupler._trial_load.size)
        self.assertEqual(
            (before["device_anderson_packet_capacity"], after["device_anderson_packet_capacity"]), (0, rows)
        )

    def check_started(self, scene):
        thread, rigid, preparation = scene.worker_thread, scene.rigid_solver, scene.coupler._host_trial_preparation
        self.assertTrue(scene._startup_ready)
        self.assertEqual((thread.graph_captures, thread.graph_launches, len(thread._step_graphs)), (2, 0, 2))
        self.assertEqual((rigid.gpu_graph_captures, rigid.gpu_graph_launches, len(rigid._rigid_graphs)), (2, 0, 2))
        self.assertEqual((preparation.captures, preparation.launches, len(preparation.graphs)), (2, 0, 2))
        coupler, solver = scene.coupler, scene.solver
        trials, held, levels = coupler._captured_trials, solver._held_start, solver.tolerance_levels()
        # The trial graph, the FEM step of the host-controlled trials and the held start: one graph
        # per buffer (and inner tolerance), launched once, with no launch or preparation counted.
        self.assertEqual((trials.captures, trials.launches, len(trials.graphs)), (2, 0, 2))
        self.assertEqual(len(levels), 5)
        self.assertEqual(
            (coupler.fem_graph_captures, coupler.fem_graph_launches, len(coupler._fem_graphs)),
            (2 * len(levels), 0, 2 * len(levels)),
        )
        self.assertEqual((held.graph_captures, held.graph_launches, held.preparations), (2, 0, 0))
        self.assertEqual(len(held._graphs), 2)
        graphs = [
            entry[0]
            for owner in (
                thread._step_graphs,
                rigid._rigid_graphs,
                preparation.graphs,
                trials.graphs,
                coupler._fem_graphs,
            )
            for entry in owner.values()
        ]
        graphs.extend(held._graphs.values())
        self.assertEqual([graph.graph_exec is None for graph in graphs], [False] * (10 + 2 * len(levels)))
        # Every FEM graph is keyed with the held start of an attempt: both of its decisions stand.
        for key, _ in coupler._fem_graphs:
            self.assertEqual(key[-1][1], (True, True))
        self.assertEqual(sorted({key[1] for key, _ in coupler._fem_graphs}), sorted(levels))
        # The workspaces of those graphs are the registered ones that the trial graph binds.
        bound = [workspace for _, workspace in trials.linear_solvers]
        for entry in coupler._fem_graphs.values():
            self.assertTrue(any(entry[2] is workspace for workspace in bound))
        self.assertEqual(solver.linear_solver_tolerance, levels[0])
        # The response-basis predictor has its storage and its column engine, and no basis.
        manager, provider = solver._affine_predictor, scene._response_basis_provider
        self.assertIsNotNone(manager.core)
        self.assertEqual((manager.active, manager.support, manager.basis_builds), (False, (), 0))
        self.assertEqual(manager.structural_key, manager._structure())
        self.assertIsNotNone(manager.capture_key())
        self.assertEqual(manager.core.enabled.numpy().tolist(), [0])
        self.assertFalse(manager.core.counters.numpy().any())
        self.assertIsNotNone(provider.engine)
        self.assertIsNotNone(provider.engine.graph.graph_exec)
        self.assertEqual((provider.reference_builds, provider.solves), (0, 0))
        # The start guess of the correction is bound to that storage and may not use it yet.
        self.assertIsNotNone(trials.rigid.guess_key)
        self.assertEqual(trials.rigid.guess.available.numpy().tolist(), [0])
        # The tetrahedron query has its hierarchy; it holds no accepted positions for the first
        # step, which is host-controlled and commits them at its acceptance.
        queries = scene.interaction._device_queries
        self.assertIsNotNone(queries.bvh)
        self.assertFalse(queries.has_previous)
        # Outputs of the certificate packet of a host trial, without and with the surface metric.
        surface = [] if coupler._surface_preconditioner is None else [coupler._surface_preconditioner.result.size + 1]
        self.assertEqual(sorted(solver._certified_linear._packets), [coupler._statistics.size + 1, *surface])
        self.assertEqual(scene._startup_tangent, (scene.solver._tangent_version, scene.sim_dt, 0))
        report = scene.report()
        census = report["capture_census"]
        self.assertEqual(census["status"], "checked")
        self.assertEqual(census["startup_unlaunched_graphs"], 0)
        # Thread 2, trial preparation 2, rigid interval 2, held start 2, FEM step 10, trial graph 2,
        # column engine 1.
        self.assertGreaterEqual(census["startup_captures"], 11 + 2 * len(levels))
        self.assertEqual(report["captures_after_startup"], 0)
        # The phase load predictor of the control frames exists, with the storage of its three frames
        # and of the engagement snapshot of every phase, and holds no history.
        predictor, phases = coupler._phase_load_predictor, round(scene.frame_dt / scene.sim_dt)
        self.assertEqual((predictor.phase_count, len(predictor.frames)), (phases, 3))
        self.assertEqual(
            (predictor.depth, predictor.attempts, predictor.commits, predictor.invalidations, predictor.frame_open),
            (0, 0, 0, 0, False),
        )
        engaged = scene.interaction._track_engaged
        for frame in predictor.frames:
            self.assertEqual(len(frame.loads), phases)
            self.assertEqual([snapshot.shape for snapshot in frame.engagement], [engaged.shape] * phases)
            self.assertEqual((frame.signatures, frame.engaged_rows), ([None] * phases, [0] * phases))
        # A scene without a trial budget has no best load of a host-controlled loop. The storage that
        # start-up makes with a budget is checked on a scene that has one
        # (test_start_up_with_a_trial_budget_makes_the_storage_of_the_best_load).
        self.assertEqual(coupler._best_load is None, getattr(coupler, "trial_budget", None) is None)
        if coupler._best_load is not None:
            self.assertEqual(coupler._best_load.shape, coupler._trial_load.shape)
        self.check_host_path_started(scene)

    def check_host_path_started(self, scene):
        """What start-up leaves for the host-controlled trials: force graphs, sample helper, correction graphs."""
        coupler, x = scene.coupler, scene.interaction
        forces, pc, surface_pc = (
            coupler._host_trial_forces,
            coupler._shaft_preconditioner,
            coupler._surface_preconditioner,
        )
        self.assertEqual(sorted(forces.graphs), _FORCE_VARIANTS)
        self.assertEqual((forces.captures, forces.launches, forces.primal_reuses, forces.fallbacks), (8, 0, 0, 0))
        # The helpers committed the tetrahedron query while the graphs were captured; it is given back.
        self.assertIsNotNone(x._device_queries.bvh)
        self.assertFalse(x._device_queries.has_previous)
        self.assertIsNone(x._previous_tip)
        self.assertIsNotNone(x._host_sample_permissions)
        replay = pc._input_preparation.replay
        self.assertEqual((replay.captures, replay.launches, replay.eager), (1, 0, 0))
        # Its solve in both forms, without and with a start guess: the response basis has its storage.
        self.assertEqual((pc.graph_captures, pc.graph_launches, pc.native_mass_fallbacks), (2, 0, 0))
        self.assertEqual(sorted(pc._graphs), [False, True])
        self.assertEqual(
            (surface_pc.graph_captures, surface_pc.graph_launches, surface_pc.native_mass_fallbacks), (1, 0, 0)
        )
        self.assertEqual((pc._native_mass_ready, surface_pc._native_mass_ready), (False, False))
        # The start guess exists with the storage of the response basis; it is attached where a
        # host-controlled correction first has an active basis, not by start-up.
        self.assertIsNotNone(pc._guess_storage)
        self.assertIsNone(pc._woodbury_guess)
        packet = pc._guess_storage._packet_preparation.replay
        self.assertEqual((packet.captures, packet.launches, packet.eager), (1, 0, 0))
        self.assertFalse(surface_pc.available)
        self.assertIsNone(x._surface_tangent_anchors)
        graphs = [
            *forces.graphs.values(),
            replay.graph,
            packet.graph,
            *(entry[1] for entry in pc._graphs.values()),
            surface_pc._graph,
        ]
        self.assertEqual([graph.graph_exec is None for graph in graphs], [False] * 13)
        workspace, rows = coupler._device_surface_anderson_workspace, min(128, coupler._trial_load.size)
        self.assertEqual((workspace._capacity, list(workspace._packets), workspace.transport_calls), (rows, [rows], 0))

    def test_start_up_call_leaves_the_scene_as_it_found_it_and_frames_replay_its_graphs(self):
        with _recorded_startup() as records, _scene(rigid_backend="mujoco-warp") as (scene, config):
            self.assertEqual(len(records), 1)
            self.check_record(scene, records[0])
            self.check_started(scene)
            thread, rigid, preparation = scene.worker_thread, scene.rigid_solver, scene.coupler._host_trial_preparation
            coupler, held, trials = scene.coupler, scene.solver._held_start, scene.coupler._captured_trials
            manager, signature = scene.solver._affine_predictor, trials.signature
            keys = (manager.capture_key(), trials.rigid.guess_key)
            host_path = (scene.coupler._host_trial_forces, scene.interaction._host_sample_permissions)
            refresh, calls = scene.solver.refresh_tangent, []

            def refreshing(particle_q, dt):
                calls.append(dt)
                return refresh(particle_q, dt)

            with patch.object(scene.solver, "refresh_tangent", side_effect=refreshing):
                scene.step()
            self.assertTrue(scene.running, scene.failure_reason)
            # The first frame used the tangent of the start-up call; only a retry would refresh.
            self.assertIsNone(scene._startup_tangent)
            if scene.step_refinements == 0:
                self.assertEqual(calls, [])
            for _ in range(2):
                scene.step()
                self.assertTrue(scene.running, scene.failure_reason)
            # The frames captured nothing at the three sites and replayed what start-up left.
            self.assertEqual(thread.graph_captures, 2)
            self.assertEqual(thread.graph_launches, thread.accepted_steps)
            self.assertEqual(thread.accepted_steps, scene.accepted_physics_steps)
            self.assertEqual((rigid.gpu_graph_captures, len(rigid._rigid_graphs)), (2, 2))
            self.assertEqual((preparation.captures, len(preparation.graphs)), (2, 2))
            self.assertIs(scene.coupler._host_trial_preparation, preparation)
            # Nor did the owners of the host-path graphs: free motion evaluates no host force graph and
            # issues no host-controlled correction, and nothing dropped a graph of start-up.
            forces, pc, surface_pc = (
                scene.coupler._host_trial_forces,
                scene.coupler._shaft_preconditioner,
                scene.coupler._surface_preconditioner,
            )
            self.assertEqual((forces.captures, sorted(forces.graphs)), (8, _FORCE_VARIANTS))
            self.assertEqual((pc.graph_captures, pc._input_preparation.replay.captures), (2, 1))
            self.assertEqual(surface_pc.graph_captures, 1)
            self.assertEqual(host_path, (forces, scene.interaction._host_sample_permissions))
            # The first interval of a run has no accepted tip and is host-controlled: it replayed.
            self.assertGreaterEqual(rigid.gpu_graph_launches, 1)
            self.assertGreaterEqual(preparation.launches, 1)
            # Its FEM steps replayed the graphs of start-up, and so did the held start of every attempt.
            self.assertEqual((coupler.fem_graph_captures, len(coupler._fem_graphs)), (10, 10))
            self.assertGreaterEqual(coupler.fem_graph_launches, 1)
            self.assertEqual(coupler.fem_graph_status, "replaying")
            self.assertEqual(held.graph_captures, 2)
            self.assertEqual(held.graph_launches, held.preparations)
            self.assertGreaterEqual(held.preparations, scene.accepted_physics_steps)
            # The device trials of the intervals after it launched the trial graphs of start-up, on
            # both buffers: no capture, no changed signature entry, the same owner.
            self.assertIs(coupler._captured_trials, trials)
            self.assertEqual((trials.captures, len(trials.graphs)), (2, 2))
            self.assertEqual(coupler.coupling_graph_captures, 2)
            self.assertGreaterEqual(coupler.coupling_graph_launches, 2)
            self.assertEqual(trials.launches, coupler.coupling_graph_launches)
            changed = [i for i, (old, new) in enumerate(zip(signature, trials.signature, strict=True)) if old != new]
            self.assertEqual(changed, [])
            self.assertEqual((manager.capture_key(), trials.rigid.guess_key), keys)
            self.assertTrue(scene.interaction._device_queries.has_previous)
            report = scene.report()
            self.assertEqual(
                (report["captures_after_startup"], report["solver_captures_after_startup"]),
                (0, 0),
                report["capture_census"],
            )
            # A reset rebuilds the scene and starts it up again.
            _reset_scene(scene, config, rebuild=True)
            self.assertEqual(len(records), 2)
            self.check_record(scene, records[1])
            self.check_started(scene)

    def test_start_up_with_a_trial_budget_makes_the_storage_of_the_best_load(self):
        # The scenes of check_started have no trial budget, and no best load. With a budget the
        # start-up call makes the storage of the best trial load of a host-controlled loop
        # (NeedleTissueCoupler.capture_graphs), so that the first such loop of a run allocates
        # nothing: the first interval of a run has no accepted tip, is host-controlled, and keeps
        # its best load in the array of start-up.
        with _scene(rigid_backend="mujoco-warp", example_arguments=("--trial-budget", "12")) as (scene, _):
            coupler = scene.coupler
            self.assertTrue(scene._startup_ready, scene._startup_status)
            self.assertEqual(coupler.trial_budget, 12)
            load, trial = coupler._best_load, coupler._trial_load
            self.assertIsInstance(load, wp.array)
            self.assertEqual((load.shape, load.dtype, load.device), (trial.shape, trial.dtype, trial.device))
            self.assertNotEqual(load.ptr, trial.ptr)
            scene.step()
            self.assertTrue(scene.running, scene.failure_reason)
            self.assertIs(coupler._best_load, load)

    def test_host_controlled_trials_and_their_correction_find_the_graphs_of_start_up(self):
        # After start-up, what a host-controlled trial loop issues is launched from the graphs of
        # start-up: its force evaluation in either recorder state and each variant, the contact
        # preparation and the solve of the shaft correction, the solve of the surface correction,
        # and the helper of the host sample mask. Nothing is captured in their files, and no build
        # of their kernel modules is loaded.
        with _scene(rigid_backend="mujoco-warp") as (scene, _):
            coupler, x, rigid, dt = scene.coupler, scene.interaction, scene.rigid_solver, scene.sim_dt
            forces, pc, surface_pc = (
                coupler._host_trial_forces,
                coupler._shaft_preconditioner,
                coupler._surface_preconditioner,
            )
            self.check_host_path_started(scene)
            kept, signatures, helper = dict(forces.graphs), dict(forces.signatures), x._host_sample_permissions
            replay = pc._input_preparation.replay
            # The solve of the shaft correction without a start guess: no basis is active here.
            correction = (replay.graph, pc._graphs[False][1], surface_pc._graph)
            sites, modules = _host_path_sites(), _host_path_modules()
            self.assertGreater(len(modules), 0)
            attempt = coupler._open_attempt(scene.state_0, scene.control, dt, None)
            attempt.predictor = None
            try:
                coupler._begin_attempt(attempt)
                coupler._candidate.assign(scene.state_0)
                for surface in (True, False):
                    surface_pc.begin_step()
                    if surface:
                        self.assertTrue(surface_pc.begin_trials(scene.state_0, scene.control, include_history=True))
                    self.assertIs(coupler._host_trial_force_graphs(), forces)
                    forces.prepare(dt)
                    self.assertEqual(forces.signature, signatures[surface])
                    for previous, fresh in ((False, True), (True, True), (False, False), (True, False)):
                        key = (surface, previous, fresh)
                        self.assertIs(forces._graph(dt, previous, fresh), kept[key], key)
                # The host sample mask, as it evaluates with a track sample.
                found, built = x._host_sample_helper()
                self.assertIs(found, helper)
                self.assertFalse(built)
                helper.prepare(x)
                helper.evaluate(scene.state_0, coupler.needle_body, certify_tip=False)
                # The two corrections, as a host-controlled trial loop prepares and issues them.
                rigid.mj_model.opt.timestep = dt / rigid.rigid_substeps
                rigid.sync_native_shadow(forward=True)
                coupler._residual.zero_()
                self.assertTrue(pc.stage(dt, tape=True))
                pc.prepare_rigid_metric(dt)
                pc._prepare_inputs(coupler._candidate, dt, coupler._residual)
                pc._solve_graph(coupler._residual, coupler._direction)
                self.assertTrue(surface_pc.begin_trials(scene.state_0, scene.control, include_history=True))
                surface_pc.active = True
                self.assertTrue(surface_pc.apply(coupler._candidate, dt, coupler._residual, coupler._direction))
            finally:
                coupler._close_attempt(attempt, scene.state_0, False)
            self.assertEqual(_host_path_sites(), sites)
            self.assertEqual(_host_path_modules(), modules)
            self.assertEqual((forces.captures, sorted(forces.graphs)), (8, _FORCE_VARIANTS))
            self.assertEqual((replay.captures, replay.launches, replay.eager), (1, 1, 0))
            self.assertEqual((pc.graph_captures, pc.graph_launches), (2, 1))
            self.assertEqual((surface_pc.graph_captures, surface_pc.graph_launches, surface_pc.applications), (1, 1, 1))
            self.assertEqual((replay.graph, pc._graph, surface_pc._graph), correction)

    def test_first_frame_has_the_tangent_of_a_scene_without_start_up(self):
        def declined(scene):
            scene._startup_status = "declined for this test"
            return False

        names = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
        with _scene(rigid_backend="mujoco-warp") as (started, _):
            with (
                patch.object(GraspScene, "capture_graphs", declined),
                _scene(rigid_backend="mujoco-warp") as (plain, _),
            ):
                self.assertTrue(started._startup_ready)
                self.assertFalse(plain._startup_ready)
                self.assertEqual(plain.report()["capture_census"]["status"], "declined for this test")
                self.assertEqual(plain.rigid_solver.gpu_graph_captures, 0)
                # Before any frame: the scene without start-up has no tangent yet.
                for scene in (started, plain):
                    scene.step()
                    self.assertTrue(scene.running, scene.failure_reason)
                    self.assertEqual(scene.step_refinements, 0)
                # One refresh each, at the same positions and time step, by the same code path.
                self.assertEqual(started.solver._tangent_version, plain.solver._tangent_version)
                self.assertEqual(_tangent(started).tobytes(), _tangent(plain).tobytes())
                self.assertEqual(started.accepted_physics_steps, plain.accepted_physics_steps)
                # The accepted states agree as two runs of one scene do on this device: its sums are not
                # ordered and the rigid constraint solve stops at its tolerance. Two scenes without
                # the call differ by up to 2.1e-6 in a velocity after this frame (measured over six
                # pairs); the bound is absolute, ten times that, with no relative part.
                for name in names:
                    velocity = name.endswith("qd")
                    np.testing.assert_allclose(
                        getattr(started.state_0, name).numpy(),
                        getattr(plain.state_0, name).numpy(),
                        rtol=0 if velocity else 3e-5,
                        atol=2e-5 if velocity else 2e-7,
                        err_msg=name,
                    )

    def test_rigid_interval_entry_ends_with_one_graph_when_its_first_interval_switches_an_option(self):
        with _scene(rigid_backend="mujoco-warp") as (scene, _):
            solver, checkpoint = scene.rigid_solver, scene.coupler._gpu_checkpoint
            self.assertEqual(len(solver._rigid_graphs), 2)
            self.assertFalse(solver.mjw_model.opt.run_rne_postconstraint)
            # A fresh solver whose output asks for body accelerations: its first interval switches
            # the backend option that computes them, which drops what that interval prepared.
            solver.invalidate_rigid_graphs()
            output = scene.model.state()
            output.body_qdd = wp.zeros(scene.model.body_count, dtype=wp.spatial_vector, device=scene.device)
            checkpoint.save()
            integration = {name: _bits(getattr(solver.mjw_data, name)) for name in INTEGRATION_FIELDS}
            counters = (solver.gpu_graph_captures, solver.gpu_graph_launches, solver.mj_model.opt.timestep)
            intervals = []

            def restore():
                intervals.append(len(solver._rigid_graphs))
                checkpoint.restore()

            solver.capture_graph(scene.state_0, output, scene.control, scene.sim_dt, before_interval=restore)
            checkpoint.restore()
            self.assertTrue(solver.mjw_model.opt.run_rne_postconstraint)
            # As written, as written again after the switch, captured and launched.
            self.assertEqual(intervals, [0, 0, 0])
            self.assertEqual(len(solver._rigid_graphs), 1)
            self.assertIsNotNone(next(iter(solver._rigid_graphs.values()))[0].graph_exec)
            self.assertEqual(
                (solver.gpu_graph_captures, solver.gpu_graph_launches, solver.mj_model.opt.timestep),
                (counters[0] + 1, counters[1], counters[2]),
            )
            for name in INTEGRATION_FIELDS:
                self.assertEqual(_bits(getattr(solver.mjw_data, name)), integration[name], name)
            self.assertTrue(np.isfinite(output.body_qdd.numpy()).all())
            # The binding has its graph: nothing more is issued.
            solver.capture_graph(scene.state_0, output, scene.control, scene.sim_dt, before_interval=self.fail)

    def test_coupler_entry_declines_without_an_attempt_and_closes_an_attempt_that_fails(self):
        with _scene(rigid_backend="mujoco-warp") as (scene, _):
            coupler, buffers = scene.coupler, (scene.state_0, scene.state_1)
            before = _startup_record(scene)
            with patch.object(coupler, "_open_attempt", side_effect=AssertionError("hidden attempt")):
                coupler.capture_fem = False
                self.assertFalse(coupler.capture_graphs(buffers, scene.control, scene.sim_dt))
                coupler.capture_fem = True
                self.assertFalse(coupler.capture_graphs((), scene.control, scene.sim_dt))
            # Every buffer has the graphs of this entry: a second call leaves the scene as it is and
            # captures none of them again but the trial graph. The scene has no accepted tip, so the
            # first call gave the tetrahedron query back without accepted positions; the second call
            # commits them again, which the trial forces report as changed storage.
            held, preparation, rigid = scene.solver._held_start, coupler._host_trial_preparation, scene.rigid_solver
            trials, fem = coupler._captured_trials, coupler.fem_graph_captures
            self.assertEqual((held.graph_captures, trials.captures), (2, 2))
            self.assertTrue(coupler.capture_graphs(buffers, scene.control, scene.sim_dt))
            self.assertEqual((preparation.captures, rigid.gpu_graph_captures, held.graph_captures), (2, 2, 2))
            self.assertEqual((preparation.launches, rigid.gpu_graph_launches, held.graph_launches), (0, 0, 0))
            self.assertEqual((coupler.fem_graph_captures, coupler.fem_graph_launches), (fem, 0))
            self.assertIs(coupler._captured_trials, trials)
            self.assertEqual((trials.captures, trials.launches, len(trials.graphs)), (4, 0, 2))
            self.assertFalse(scene.interaction._device_queries.has_previous)
            after = _startup_record(scene)
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[1], before[1])
            before[2]["captures_after_startup"] += 2
            before[2]["coupling_graph_captures"] += 2
            self.assertEqual(_flat(after[2]), _flat(before[2]))
            # An error inside an attempt: the attempt is closed like a rejected step, the counters are back.
            closed = []
            close = coupler._close_attempt

            def closing(attempt, state_in, committed):
                closed.append((state_in, committed))
                return close(attempt, state_in, committed)

            with (
                patch.object(coupler, "_close_attempt", side_effect=closing),
                patch.object(coupler._shaft_preconditioner, "begin_step", side_effect=RuntimeError("begin failed")),
                self.assertRaisesRegex(RuntimeError, "begin failed"),
            ):
                coupler.capture_graphs(buffers, scene.control, scene.sim_dt)
            self.assertEqual(closed, [(scene.state_0, False)])
            after = _startup_record(scene)
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[1], before[1])
            self.assertEqual(_flat(after[2]), _flat(before[2]))
            scene.step()
            self.assertTrue(scene.running, scene.failure_reason)


class GraspSceneCensusTests(unittest.TestCase):
    """After start-up a scene on the MuJoCo Warp backend captures no CUDA graph and loads no kernel module."""

    def setUp(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("The start-up call captures CUDA graphs")

    def check_nothing_late(self, scene, begun):
        """The scene's own numbers after a walk of the kernel modules, and the captures Warp began."""
        scene.refresh_capture_census()
        report = scene.report()
        census = report["capture_census"]
        self.assertEqual(census["status"], "checked")
        self.assertEqual(_late(report), _NOTHING_LATE, census)
        self.assertEqual((census["late_sites"], census["late_modules"]), ({}, {}))
        self.assertEqual(census["graphs_released_after_startup"], 0)
        self.assertEqual(begun, [])
        return report

    def test_default_procedure_captures_and_loads_nothing_after_start_up(self):
        self.procedure((), 200)

    def test_procedure_with_host_decided_surface_events_reaches_the_host_path_graphs_and_captures_nothing(self):
        # With the demo's defaults a surface event is found by the device trials and decided on the
        # last of them, and the drive above launches no graph of a host-controlled trial loop in the
        # tissue. With the event options at their off values the host decides a surface event from a
        # complete trial of its own: the second puncture (frame 212 of this drive) is a
        # host-controlled loop with captured forces and a correction. Start-up gave the counters of
        # those graph sites back, so a capture at one of them would be seen here.
        def reached(scene, report, marked):
            launched = (
                report["host_force_graph_launches"],
                report["contact_preparation_launches"],
                scene.coupler._shaft_preconditioner.graph_launches,
            )
            # Host force graph, contact preparation, shaft solve: each was launched after the mark.
            moved = [count > before for count, before in zip(launched, marked, strict=True)]
            self.assertEqual(moved, [True, True, True], (launched, marked))

        self.procedure(
            ("--event-search", "tip-path", "--event-oracle", "restart", "--response-basis-nodes", "0"), 240, reached
        )

    def procedure(self, options, frames, reached=None):
        # The scene of the demo, built in this process and driven as the example drives it: free
        # motion, first contact, puncture and a growing track. One step of an interval in the
        # tissue is rejected, which halves the interval and makes the number of accepted
        # intervals odd, so that frames start from either accepted-state buffer. ``reached`` is
        # called after the frames with the scene, its report and the launch counters of the
        # host-path graph sites at the mark (host force graph, contact preparation, shaft solve).
        from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example
        from newton_solver_suturebot.examples.grasp._grasp_procedure import ProceduralGraspController

        defaults = grasp_example.Example.create_parser().parse_args(list(options))
        config = SutureSceneConfig(
            device=os.environ["NEWTON_TEST_DEVICE"],
            control_mode="physical_grasp",
            rigid_backend=defaults.rigid_backend,
            substeps=defaults.substeps,
            example_arguments=grasp_example.Example._coupling(defaults),
            **grasp_example.Example._bite(defaults.bite_length),
        )
        self.assertEqual(config.rigid_backend, "mujoco-warp")
        rejected_frame = 120
        scene = _make_scene(config, 0)
        try:
            # As a worker: the first observation, then the mark that ends start-up.
            observation = _snapshot(_observe(scene, 0))
            controller = ProceduralGraspController(
                _snapshot(_geometry(scene, config)),
                config,
                defaults.speed,
                defaults.settle_seconds,
                lift_height=defaults.lift_height,
                tip_lift_height=grasp_example.Example._tip_lift_height(defaults),
                pull_through=grasp_example.Example._pull_through(defaults),
            )
            _mark_startup(scene)
            report = scene.report()
            census = report["capture_census"]
            self.assertEqual(census["status"], "checked")
            self.assertEqual(census["startup_unlaunched_graphs"], 0)
            self.assertGreater(census["startup_captures"], 0)
            # Storage that start-up made for the frames: the engagement snapshots of the phase load
            # predictor and, with the trial budget of the demo, the best load of a host loop.
            predictor, best_load = scene.coupler._phase_load_predictor, scene.coupler._best_load
            snapshots = sorted(array.ptr for frame in predictor.frames for array in frame.engagement)
            self.assertEqual(len(snapshots), 3 * config.substeps)
            self.assertEqual(best_load.shape, scene.coupler._trial_load.shape)
            marked = (
                report["host_force_graph_launches"],
                report["contact_preparation_launches"],
                scene.coupler._shaft_preconditioner.graph_launches,
            )
            check, attempts, odd_ends, late_frames = scene.coupler._tet_step_guard.check, [], 0, {}

            def once(state_in, candidate):
                attempts.append(scene.accepted_physics_steps)
                # The last interval of the frame: its candidate is refused once.
                if len(attempts) == config.substeps:
                    raise NeedleStepTooLarge("refused by the test")
                return check(state_in, candidate)

            with _counted_captures() as begun:
                for frame in range(frames):
                    action = controller([observation])
                    self.assertNotEqual(controller.report()[0]["phase"], "stopped", controller.report())
                    _apply_action(scene, config, action[0])
                    if frame == rejected_frame:
                        refinements, rejections = scene.step_refinements, scene.coupler.tet_step_rejections
                        with patch.object(scene.coupler._tet_step_guard, "check", side_effect=once):
                            body_q = _step(scene)
                        self.assertGreaterEqual(scene.coupler.tet_step_rejections, rejections + 1)
                        self.assertGreater(scene.step_refinements, refinements)
                        # The refused interval was attempted with a track in the tissue.
                        self.assertGreater(scene.interaction.track_count, 0)
                    else:
                        body_q = _step(scene)
                    self.assertTrue(scene.running, scene.failure_reason)
                    observation = _snapshot(
                        _observe(scene, 0, include_tissue=False, include_thread_diagnostics=False, body_q=body_q)
                    )
                    scene.actual_tip_history.clear()
                    odd_ends += scene.accepted_physics_steps % 2
                    # The counters of every frame: a capture is named with the frame it was made in.
                    counted = (*(observation["diagnostics"][name] for name in _LATE_NUMBERS[:2]), len(begun))
                    if counted != late_frames.get("last", (0, 0, 0)):
                        late_frames[frame] = late_frames["last"] = counted
                self.assertEqual(late_frames, {}, (begun, scene.report()["capture_census"]))
                report = self.check_nothing_late(scene, begun)
            if reached is not None:
                reached(scene, report, marked)
            # The predictor took its engagement snapshots in the storage of start-up, and the
            # budgeted host loops kept their best load in the array of start-up.
            self.assertIs(scene.coupler._phase_load_predictor, predictor)
            self.assertIs(scene.coupler._best_load, best_load)
            self.assertEqual(sorted(array.ptr for frame in predictor.frames for array in frame.engagement), snapshots)
            self.assertTrue(any(rows for frame in predictor.frames for rows in frame.engaged_rows))
            # What the frames went through: the needle entered the tissue and left samples behind,
            # in storage that did not grow, and frames ended on either buffer.
            self.assertGreaterEqual(report["entry_count"], 1)
            self.assertGreater(report["track_samples"], 1)
            self.assertGreaterEqual(report["openings"], 1)
            self.assertEqual([report[name] for name in ("track_storage_growths", "opening_storage_growths")], [0, 0])
            self.assertEqual(report["shaft_pc_storage_growths"], 0)
            self.assertGreater(odd_ends, 0)
            self.assertLess(odd_ends, frames)
            self.assertGreater(scene.accepted_physics_steps, frames * config.substeps)
        finally:
            scene.close()
            scene.viewer.close()

    @contextmanager
    def local_scene(self):
        """A scene in this process behind the interface of the workers, on the small mesh of this module."""
        with (
            TemporaryDirectory() as directory,
            patch.object(phantom, "list_mesh_regions", return_value=_REFERENCE.regions),
            patch.object(phantom, "load_mesh_regions", side_effect=_selection),
        ):
            config = SutureSceneConfig(
                device=os.environ["NEWTON_TEST_DEVICE"],
                asset=str(Path(directory) / "pad.usd"),
                entry_xy=(0, 0.005),
                exit_xy=(0, -0.005),
                control_mode="physical_grasp",
                rigid_backend="mujoco-warp",
            )
            with LocalSutureScene(config) as scenes:
                yield scenes, config

    def test_local_scene_reports_nothing_late_after_frames_through_its_interface(self):
        with self.local_scene() as (scenes, config), _counted_captures() as begun:
            hold = np.zeros((1, config.action_dimension))
            for _ in range(20):
                stepped = scenes.step(hold, include_tissue=False, include_thread_diagnostics=False)[0]["diagnostics"]
                self.assertIsNone(stepped.get("failure_reason"))
            # A step reports the capture counters as they are and the module numbers of the last
            # observation; an explicit observation walks the kernel modules again.
            self.assertEqual(stepped["capture_census"]["status"], "checked")
            self.assertEqual(_late(stepped), _NOTHING_LATE, stepped["capture_census"])
            observed = scenes.observe()[0]["diagnostics"]
            census = observed["capture_census"]
            self.assertEqual(census["status"], "checked")
            self.assertEqual(_late(observed), _NOTHING_LATE, census)
            self.assertEqual((census["late_sites"], census["late_modules"]), ({}, {}))
            self.assertEqual((census["startup_unlaunched_graphs"], census["graphs_released_after_startup"]), (0, 0))
            self.assertEqual(observed["accepted_physics_steps"], 20 * config.substeps)
            self.assertEqual(begun, [])

    def test_reset_starts_up_again_and_its_frames_capture_and_load_nothing(self):
        with self.local_scene() as (scenes, config):
            hold = np.zeros((1, config.action_dimension))
            first = scenes.observe()[0]["diagnostics"]["capture_census"]
            self.assertEqual(first["status"], "checked")
            # The scene has run before it is rebuilt: both accepted-state buffers were used.
            for _ in range(3):
                scenes.step(hold, include_tissue=False, include_thread_diagnostics=False)
            with _counted_captures() as during_reset:
                reset = scenes.reset()[0]
            self.assertEqual(reset["reset_count"], 1)
            second = reset["diagnostics"]["capture_census"]
            # The census describes the start-up of this reset: as many captures as the first, each
            # launched, and every one of them counted where Warp begins a capture.
            self.assertEqual(second["status"], "checked")
            self.assertEqual(
                (second["startup_captures"], second["startup_solver_captures"]),
                (first["startup_captures"], first["startup_solver_captures"]),
            )
            self.assertEqual(second["startup_captures"] + second["startup_solver_captures"], len(during_reset))
            self.assertEqual(second["startup_unlaunched_graphs"], 0)
            self.assertEqual(_late(reset["diagnostics"]), _NOTHING_LATE, second)
            with _counted_captures() as begun:
                for _ in range(20):
                    stepped = scenes.step(hold, include_tissue=False, include_thread_diagnostics=False)[0]
                    self.assertIsNone(stepped["diagnostics"].get("failure_reason"))
                observed = scenes.observe()[0]["diagnostics"]
            census = observed["capture_census"]
            self.assertEqual(census["status"], "checked")
            self.assertEqual(_late(observed), _NOTHING_LATE, census)
            self.assertEqual((census["late_sites"], census["late_modules"]), ({}, {}))
            self.assertEqual(census["graphs_released_after_startup"], 0)
            self.assertEqual(observed["accepted_physics_steps"], 20 * config.substeps)
            self.assertEqual(begun, [])


if __name__ == "__main__":
    unittest.main()
