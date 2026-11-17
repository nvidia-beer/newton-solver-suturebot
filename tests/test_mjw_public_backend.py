"""Public backend selection and the explicit FP32 rigid stopping contract."""

import dataclasses
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco_warp import (
    INTEGRATION_FIELDS,
    SubcycledMuJoCoWarp,
    WarpIntegrationCheckpoint,
    _DeviceArrayIndex,
)
from newton_solver_suturebot.examples.grasp.parallel import SutureSceneConfig


class PublicBackendTests(unittest.TestCase):
    def test_package_imports_without_mujoco_and_mujoco_warp(self):
        # Both belong to the examples extra. The GPU backend imports MuJoCo Warp when it first steps.
        code = "import sys, newton_solver_suturebot; print([name in sys.modules for name in ('mujoco', 'mujoco_warp')])"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines()[-1], "[False, False]")

    def test_native_default_and_explicit_gpu_scope(self):
        self.assertEqual(SutureSceneConfig(device="cpu").rigid_backend, "native")
        config = SutureSceneConfig(device="cuda:0", control_mode="physical_grasp", rigid_backend="mujoco-warp")
        self.assertEqual(config.rigid_backend, "mujoco-warp")
        for changes in ({"rigid_backend": "unknown"}, {"device": "cpu"}, {"control_mode": "free_needle"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SutureSceneConfig(
                    **({"device": "cuda:0", "control_mode": "physical_grasp", "rigid_backend": "mujoco-warp"} | changes)
                )

    def test_gpu_precision_is_a_separate_audited_contract_not_a_native_equality_claim(self):
        solver = object.__new__(SubcycledMuJoCoWarp)
        solver.gpu_tolerance = 1e-6
        native = dict(iterations=200, ls_iterations=50, ls_tolerance=0.01, integrator=3, cone=1, solver=2)
        solver.mj_model = SimpleNamespace(opt=SimpleNamespace(**native, tolerance=1e-12, impratio=10.0))
        effective = dict(native)
        effective["tolerance"] = wp.array([1e-6], dtype=wp.float32, device="cpu")
        effective["ls_tolerance"] = wp.array([0.01], dtype=wp.float32, device="cpu")
        effective["impratio_invsqrt"] = wp.array([1.0 / np.sqrt(10.0)], dtype=wp.float32, device="cpu")
        solver.mjw_model = SimpleNamespace(opt=SimpleNamespace(**effective))
        options = solver.audit_gpu_options()
        self.assertEqual(options["tolerance"], [float(np.float32(1e-6))])
        self.assertEqual(solver.mj_model.opt.tolerance, 1e-12)
        solver.mjw_model.opt.tolerance.fill_(1e-5)
        with self.assertRaisesRegex(AssertionError, "explicit GPU precision"):
            solver.audit_gpu_options()
        solver.mjw_model.opt.tolerance.fill_(1e-6)
        solver.mjw_model.opt.iterations = 199
        with self.assertRaisesRegex(AssertionError, "iterations"):
            solver.audit_gpu_options()

    def test_solver_budget_flags_are_not_reported_as_contact_capacity_only(self):
        solver = object.__new__(SubcycledMuJoCoWarp)
        solver.mjw_data = SimpleNamespace(overflow=wp.zeros(1, dtype=int, device="cpu"))
        solver.check_overflow()
        for flag, name in ((512, "ITERATIONS"), (1024, "LS_ITERATIONS"), (8, "NARROWPHASE"), (1 << 20, "UNKNOWN")):
            solver.mjw_data.overflow.fill_(flag)
            with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, name):
                solver.check_overflow()

    def test_unstable_rigid_trial_is_rejected_like_a_native_bad_state_warning(self):
        import mujoco

        from newton_solver_suturebot import NeedleStepTooLarge
        from newton_solver_suturebot._src.solvers.mujoco.subcycled_mujoco_warp import WarpRigidStepRejected

        solver = object.__new__(SubcycledMuJoCoWarp)
        solver._mujoco = mujoco
        solver.gpu_guard_rejections = 0
        fields = {name: np.full((1, 3), 0.25, dtype=np.float32) for name in ("qpos", "qvel", "qacc")}

        def data():
            return SimpleNamespace(**{name: wp.array(values, device="cpu") for name, values in fields.items()})

        solver.mjw_data = data()
        solver.reject_unstable_trial()
        self.assertEqual(solver.gpu_guard_rejections, 0)
        # MuJoCo's own limit, not only non-finite values.
        for name, value in (("qvel", np.nan), ("qpos", np.inf), ("qacc", 2 * mujoco.mjMAXVAL)):
            with self.subTest(field=name, value=value):
                original = fields[name].copy()
                fields[name][0, 1] = value
                solver.mjw_data = data()
                rejections = solver.gpu_guard_rejections
                with self.assertRaisesRegex(WarpRigidStepRejected, name):
                    solver.reject_unstable_trial()
                self.assertEqual(solver.gpu_guard_rejections, rejections + 1)
                fields[name] = original
        # The enclosing step refines and retries on this exception type.
        self.assertTrue(issubclass(WarpRigidStepRejected, NeedleStepTooLarge))

    def test_checkpoint_rollback_invalidates_prepared_clock_witness(self):
        from unittest.mock import Mock

        solver = object.__new__(SubcycledMuJoCoWarp)
        solver.model = SimpleNamespace(device=wp.get_device("cpu"), requires_grad=False)
        solver.use_mujoco_cpu = solver.enable_sleeping = False
        solver.mj_model = SimpleNamespace(nplugin=0)
        solver.mjw_data = SimpleNamespace(
            nworld=1, **{name: wp.zeros((1, 1), device="cpu") for name in INTEGRATION_FIELDS}
        )
        solver.mjw_data.time = wp.zeros(1, device="cpu")
        solver.rigid_substeps = 4
        solver._validate_execution_context = Mock()
        solver.check_overflow = Mock()
        solver._grasp_body_ids = None
        solver._accepted_clock = np.zeros(1, dtype=np.float32)
        solver._actual_accepted_clock = solver._accepted_clock.copy()
        solver._accepted_intervals = np.zeros(1, dtype=np.int64)
        solver._accepted_generation = 0
        solver._prepared_observations = None
        solver._trial_dt = 0.01
        checkpoint = WarpIntegrationCheckpoint(solver)
        checkpoint.save()
        expected = np.float32(0)
        for _ in range(4):
            expected = np.float32(expected + np.float32(0.01 / 4))
        solver.mjw_data.time.assign([expected])
        witness = solver.prepare_observations()
        checkpoint.restore()
        with self.assertRaisesRegex(ValueError, "single-use"):
            solver.commit_observations(prepared=witness)
        np.testing.assert_array_equal(solver._accepted_clock, [0])
        np.testing.assert_array_equal(solver._accepted_intervals, [0])
        solver.mjw_data.time.assign([expected])
        solver.commit_observations(prepared=solver.prepare_observations())
        np.testing.assert_array_equal(solver._accepted_intervals, [1])

    def test_array_index_notices_every_replaced_array_and_container(self):
        @dataclasses.dataclass
        class Inner:
            values: wp.array
            count: int

        @dataclasses.dataclass
        class Outer:
            first: wp.array
            inner: Inner
            levels: tuple
            tiles: tuple
            name: str
            callback: object

        def array():
            return wp.zeros(2, device="cpu")

        def owner():
            return Outer(array(), Inner(array(), 3), (array(), array()), (Inner(array(), 1),), "model", print)

        model, data = owner(), owner()
        index = _DeviceArrayIndex(model, data)
        self.assertEqual(len(index.arrays), 10)
        self.assertEqual(
            [id(value) for value in index.arrays[:5]],
            [id(value) for value in (model.first, model.inner.values, *model.levels, model.tiles[0].values)],
        )
        self.assertTrue(index.unchanged())
        # what is not device storage is not followed
        model.name, model.callback, model.inner.count = "other", len, 4
        self.assertTrue(index.unchanged())
        replacements = (
            lambda: setattr(model, "first", array()),
            lambda: setattr(data.inner, "values", array()),
            lambda: setattr(model, "inner", Inner(model.inner.values, 3)),
            lambda: setattr(data, "levels", (data.levels[0], array())),
            lambda: setattr(model.tiles[0], "values", array()),
            lambda: setattr(data, "tiles", (Inner(data.tiles[0].values, 1),)),
        )
        for number, replace in enumerate(replacements):
            with self.subTest(replacement=number):
                replace()
                self.assertFalse(index.unchanged())
                index = _DeviceArrayIndex(model, data)
                self.assertTrue(index.unchanged())
        model.levels = [array()]
        with self.assertRaisesRegex(TypeError, "dataclasses and tuples"):
            _DeviceArrayIndex(model)

    def _interval_double(self, *, switches=0, never=False):
        """A solver whose ``step`` follows the rule of the real one, without a device.

        The first interval of a warmed-up state runs as written, the next interval of a binding
        is captured and launched. ``switches`` intervals run as written and switch a scalar option,
        which the next preflight answers by dropping what was prepared (``invalidate_rigid_graphs``).
        """
        solver = object.__new__(SubcycledMuJoCoWarp)
        solver.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        solver.rigid_substeps = 4
        solver._rigid_graphs = {}
        solver.gpu_graph_captures = solver.gpu_graph_launches = 0
        log = SimpleNamespace(intervals=[], warmed=False, generation=0, pending=False, switches=switches)

        def validate(state_in, state_out, control, dt):
            if log.pending:
                log.pending, log.warmed = False, False
                log.generation += 1
                solver._rigid_graphs.clear()
            return (log.generation, state_in, state_out)

        def step(state_in, state_out, control, contacts, dt):
            key = validate(state_in, state_out, control, dt)
            solver.mj_model.opt.timestep = dt / solver.rigid_substeps
            if not log.warmed:
                log.intervals.append("as written")
                log.warmed = True
                if log.switches:
                    log.switches -= 1
                    log.pending = True
                return
            if key not in solver._rigid_graphs and not never:
                log.intervals.append("captured")
                solver._rigid_graphs[key] = object()
                solver.gpu_graph_captures += 1
            else:
                log.intervals.append("launched")
            solver.gpu_graph_launches += 1

        solver.validate_capture_bindings, solver.step = validate, step
        return solver, log

    def test_capture_graph_issues_the_intervals_that_end_with_one_graph_per_binding(self):
        solver, log = self._interval_double()
        solver.gpu_graph_launches = 7
        restores = []
        solver.capture_graph(
            "in 0", "out", "control", 0.004, before_interval=lambda: restores.append(len(log.intervals))
        )
        # The first binding of a solver: one interval as written, one captured and launched.
        self.assertEqual(log.intervals, ["as written", "captured"])
        # The caller's restore runs before each interval.
        self.assertEqual(restores, [0, 1])
        self.assertEqual((solver.gpu_graph_captures, solver.gpu_graph_launches), (1, 7))
        self.assertEqual(solver.mj_model.opt.timestep, 0.002)
        # A second binding of the warmed solver takes one interval; one that has its graph takes none.
        solver.capture_graph(
            "in 1", "out", "control", 0.004, before_interval=lambda: restores.append(len(log.intervals))
        )
        self.assertEqual(log.intervals, ["as written", "captured", "captured"])
        self.assertEqual(restores, [0, 1, 2])
        for binding in ("in 0", "in 1"):
            solver.capture_graph(binding, "out", "control", 0.004, before_interval=self.fail)
        self.assertEqual(len(log.intervals), 3)
        self.assertEqual((len(solver._rigid_graphs), solver.gpu_graph_captures, solver.gpu_graph_launches), (2, 2, 7))
        # Without a restore of the caller's the intervals are issued all the same.
        solver.capture_graph("in 2", "out", "control", 0.004)
        self.assertEqual((len(solver._rigid_graphs), solver.gpu_graph_launches), (3, 7))

    def test_capture_graph_covers_an_option_that_the_first_interval_switches(self):
        solver, log = self._interval_double(switches=1)
        solver.capture_graph("in 0", "out", "control", 0.004)
        # The switch drops what the first interval prepared: one more as written, then the capture.
        self.assertEqual(log.intervals, ["as written", "as written", "captured"])
        self.assertEqual((len(solver._rigid_graphs), solver.gpu_graph_captures, solver.gpu_graph_launches), (1, 1, 0))
        solver.capture_graph("in 1", "out", "control", 0.004)
        self.assertEqual((len(solver._rigid_graphs), solver.gpu_graph_captures, solver.gpu_graph_launches), (2, 2, 0))

    def test_capture_graph_raises_after_three_intervals_without_a_graph_and_keeps_its_counters(self):
        for options in ({"switches": 3}, {"never": True}):
            with self.subTest(options=options):
                solver, log = self._interval_double(**options)
                solver.gpu_graph_launches = 5
                with self.assertRaisesRegex(RuntimeError, "no graph after three intervals"):
                    solver.capture_graph("in 0", "out", "control", 0.004)
                self.assertEqual(len(log.intervals), 3)
                self.assertEqual(solver.gpu_graph_launches, 5)
                self.assertEqual(solver.mj_model.opt.timestep, 0.002)
        # An error of an interval leaves the counters as they were, too.
        solver, log = self._interval_double()
        solver.gpu_graph_launches = 5

        def failing(*arguments):
            solver.gpu_graph_launches += 1
            solver.mj_model.opt.timestep = 0.001
            raise FloatingPointError("interval failed")

        solver.step = failing
        with self.assertRaisesRegex(FloatingPointError, "interval failed"):
            solver.capture_graph("in 0", "out", "control", 0.004)
        self.assertEqual((solver.gpu_graph_launches, solver.mj_model.opt.timestep), (5, 0.002))

    def test_invalid_gpu_tolerance_fails_before_model_or_device_initialization(self):
        for tolerance in (0.0, -1.0, np.inf, np.nan, True, "1e-6", None):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "gpu_tolerance"):
                SubcycledMuJoCoWarp(None, gpu_tolerance=tolerance)


if __name__ == "__main__":
    unittest.main()
