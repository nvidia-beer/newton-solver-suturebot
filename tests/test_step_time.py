# SPDX-License-Identifier: Apache-2.0
"""The time step as device data: exact values, guarded writes and capture behaviour."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot._src.utils.step_time import StepTime

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
# The base step of the grasp scene, its halvings, a frame remainder that differs from
# a halving in its last bits, and a step that is not representable in float32.
STEPS = (1.0 / 240.0, 1.0 / 480.0, 0.0020833333333333355, 1.0 / 960.0, 0.0005, 1.0e-4 / 3.0)


@wp.kernel(enable_backward=False)
def _scaled_by_argument(source: wp.array[wp.vec3], dt: float, target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] = target[i] + dt * source[i]


@wp.kernel(enable_backward=False)
def _scaled_by_array(source: wp.array[wp.vec3], dt: wp.array[float], target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] = target[i] + dt[0] * source[i]


@wp.kernel(enable_backward=False)
def _double_by_argument(source: wp.array[wp.float64], dt: wp.float64, target: wp.array[wp.float64]):
    i = wp.tid()
    target[i] = target[i] + dt * source[i]


@wp.kernel(enable_backward=False)
def _double_by_array(source: wp.array[wp.float64], dt: wp.array[wp.float64], target: wp.array[wp.float64]):
    i = wp.tid()
    target[i] = target[i] + dt[0] * source[i]


class StepTimeTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = wp.get_device(DEVICE)

    def test_arrays_hold_the_values_the_host_formed_in_double_precision(self):
        step_time = StepTime(self.device)
        quarter = step_time.fraction(4)
        self.assertIsNone(step_time.value)
        for dt in STEPS:
            step_time.set(dt)
            self.assertEqual(step_time.value, dt)
            # What a launch argument of the former parameter type would have carried.
            np.testing.assert_array_equal(step_time.seconds.numpy(), np.array([dt], dtype=np.float32))
            np.testing.assert_array_equal(step_time.seconds_f64.numpy(), np.array([dt], dtype=np.float64))
            np.testing.assert_array_equal(quarter.numpy(), np.array([dt / 4], dtype=np.float32))
        self.assertEqual(step_time.seconds.dtype, wp.float32)
        self.assertEqual(step_time.seconds_f64.dtype, wp.float64)
        self.assertEqual(quarter.dtype, wp.float32)
        for array in (step_time.seconds, step_time.seconds_f64, quarter):
            self.assertEqual(array.shape, (1,))
            self.assertFalse(array.requires_grad)

    def test_only_a_changed_value_is_written_and_storage_is_never_replaced(self):
        step_time = StepTime(self.device)
        quarter = step_time.fraction(4)
        pointers = (step_time.seconds.ptr, step_time.seconds_f64.ptr, quarter.ptr)
        writes = []
        for dt in (0.004, 0.004, 0.002, 0.002, 0.004, np.float64(0.004), 0.004):
            step_time.set(dt)
            writes.append(step_time.writes)
        self.assertEqual(writes, [1, 1, 2, 2, 3, 3, 3])
        self.assertIs(step_time.fraction(4), quarter)
        self.assertEqual(pointers, (step_time.seconds.ptr, step_time.seconds_f64.ptr, quarter.ptr))
        # A fraction requested after a write holds the current value at once.
        third = step_time.fraction(3)
        np.testing.assert_array_equal(third.numpy(), np.array([0.004 / 3], dtype=np.float32))
        step_time.set(0.001)
        np.testing.assert_array_equal(third.numpy(), np.array([0.001 / 3], dtype=np.float32))
        np.testing.assert_array_equal(quarter.numpy(), np.array([0.001 / 4], dtype=np.float32))
        for parts in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                step_time.fraction(parts)
        # Any other value the host derives from dt is held the same way, in the type it is read as.
        inverse = step_time.derived("inverse", lambda dt: 1.0 / dt, wp.float64)
        np.testing.assert_array_equal(inverse.numpy(), np.array([1.0 / 0.001]))
        step_time.set(1.0 / 240.0)
        np.testing.assert_array_equal(inverse.numpy(), np.array([1.0 / (1.0 / 240.0)]))
        self.assertIs(step_time.derived("inverse", lambda dt: 1.0 / dt, wp.float64), inverse)
        with self.assertRaises(ValueError):
            step_time.derived("inverse", lambda dt: 1.0 / dt, wp.float32)

    def test_value_with_host_parameters_keeps_one_array_and_follows_them(self):
        # A value the host forms from the time step and another setting, such as the thread's
        # 1 / dt² + c / dt with its mass damping c: one array for the life of the object, refilled
        # when the setting or the time step changes. An array per value of the setting would
        # grow without bound under a schedule of that setting.
        step_time = StepTime(self.device)

        def scale(damping):
            return step_time.derived(
                "inertia scale", lambda dt: 1.0 / (dt * dt) + damping / dt, wp.float64, parameters=damping
            )

        array = scale(2.0)
        pointer = array.ptr
        np.testing.assert_array_equal(array.numpy(), [0.0])  # no time step yet
        step_time.set(1.0 / 240.0)
        np.testing.assert_array_equal(array.numpy(), [1.0 / ((1.0 / 240.0) * (1.0 / 240.0)) + 2.0 / (1.0 / 240.0)])
        for damping in (5.0, 0.0, 117.72, 2.0, 2.0):
            self.assertIs(scale(damping), array)
            dt = step_time.value
            np.testing.assert_array_equal(array.numpy(), [1.0 / (dt * dt) + damping / dt])
            # The time step changes: the array follows with the setting requested last.
            step_time.set(dt / 2)
            np.testing.assert_array_equal(array.numpy(), [1.0 / ((dt / 2) * (dt / 2)) + damping / (dt / 2)])
        self.assertEqual((array.ptr, len(step_time._derived)), (pointer, 1))
        # The same setting again is no write.
        with patch.object(type(array), "fill_", side_effect=AssertionError("written")):
            self.assertIs(scale(2.0), array)
        with self.assertRaises(ValueError):
            step_time.derived("inertia scale", lambda dt: dt, wp.float32, parameters=2.0)

    def test_failed_write_leaves_the_arrays_and_the_value_as_they_were(self):
        # The thread's value 1 / dt² + c / dt raises for a time step of zero. The rejected time
        # step must not reach the device: a later step at the previous one writes nothing, so it
        # would read the rejected value.
        step_time = StepTime(self.device)
        quarter = step_time.fraction(4)
        inverse = step_time.derived("inverse", lambda dt: 1.0 / (dt * dt) + 2.0 / dt, wp.float64)
        dt = 1.0 / 240.0
        step_time.set(dt)

        def contents():
            return [array.numpy().copy() for array in (step_time.seconds, step_time.seconds_f64, quarter, inverse)]

        before = contents()
        with self.assertRaises(ZeroDivisionError):
            step_time.set(0.0)
        self.assertEqual((step_time.value, step_time.writes), (dt, 1))
        for held, expected in zip(contents(), before, strict=True):
            np.testing.assert_array_equal(held, expected)
        step_time.set(dt)
        self.assertEqual(step_time.writes, 1)
        # A value requested with an expression that raises leaves no entry behind.
        with self.assertRaises(ZeroDivisionError):
            step_time.derived("broken", lambda dt: 1.0 / (dt - dt))
        self.assertNotIn("broken", step_time._derived)
        # A changed setting whose expression raises keeps the array and its expression.
        with self.assertRaises(ZeroDivisionError):
            step_time.derived("inverse", lambda dt: 1.0 / (dt - dt), wp.float64, parameters="other")
        np.testing.assert_array_equal(inverse.numpy(), before[3])
        step_time.set(dt / 2)
        np.testing.assert_array_equal(inverse.numpy(), [1.0 / ((dt / 2) * (dt / 2)) + 2.0 / (dt / 2)])

        # A write that stops between two arrays (an interrupt, a device error): the value is
        # unknown, and the next set writes every array even for the value held before.
        dt = step_time.value
        fill = type(step_time.seconds).fill_
        calls = []

        def interrupted(array, value):
            calls.append(array.ptr)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return fill(array, value)

        with patch.object(type(step_time.seconds), "fill_", interrupted), self.assertRaises(KeyboardInterrupt):
            step_time.set(0.003)
        self.assertEqual(calls, [step_time.seconds.ptr, step_time.seconds_f64.ptr])
        self.assertIsNone(step_time.value)
        np.testing.assert_array_equal(step_time.seconds.numpy(), [np.float32(0.003)])
        np.testing.assert_array_equal(step_time.seconds_f64.numpy(), [dt])
        writes = step_time.writes
        step_time.set(dt)
        self.assertEqual((step_time.value, step_time.writes), (dt, writes + 1))
        np.testing.assert_array_equal(step_time.seconds.numpy(), [np.float32(dt)])
        np.testing.assert_array_equal(step_time.seconds_f64.numpy(), [dt])
        np.testing.assert_array_equal(quarter.numpy(), [np.float32(dt / 4)])
        np.testing.assert_array_equal(inverse.numpy(), [1.0 / (dt * dt) + 2.0 / dt])

    def test_kernels_reading_the_array_repeat_the_bits_of_a_launch_argument(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(257, 3)).astype(np.float32)
        doubles = rng.normal(size=257)
        source = wp.array(values, dtype=wp.vec3, device=self.device)
        source_f64 = wp.array(doubles, dtype=wp.float64, device=self.device)
        step_time = StepTime(self.device)
        for dt in STEPS:
            step_time.set(dt)
            by_argument = wp.array(values[::-1].copy(), dtype=wp.vec3, device=self.device)
            by_array = wp.array(values[::-1].copy(), dtype=wp.vec3, device=self.device)
            wp.launch(_scaled_by_argument, len(values), [source, dt, by_argument], device=self.device)
            wp.launch(_scaled_by_array, len(values), [source, step_time.seconds, by_array], device=self.device)
            np.testing.assert_array_equal(by_array.numpy(), by_argument.numpy())
            by_argument = wp.array(doubles[::-1].copy(), dtype=wp.float64, device=self.device)
            by_array = wp.array(doubles[::-1].copy(), dtype=wp.float64, device=self.device)
            wp.launch(_double_by_argument, len(doubles), [source_f64, dt, by_argument], device=self.device)
            wp.launch(_double_by_array, len(doubles), [source_f64, step_time.seconds_f64, by_array], device=self.device)
            np.testing.assert_array_equal(by_array.numpy(), by_argument.numpy())

    def test_write_during_capture_is_refused_and_a_captured_reader_follows_later_writes(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture")
        values = np.arange(12, dtype=np.float32).reshape(4, 3)
        source = wp.array(values, dtype=wp.vec3, device=self.device)
        target = wp.zeros(4, dtype=wp.vec3, device=self.device)
        step_time = StepTime(self.device)
        step_time.set(0.004)
        scaled = step_time.derived("scaled", lambda dt: 3.0 * dt, parameters=3.0)
        wp.launch(_scaled_by_array, 4, [source, step_time.seconds, target], device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            # The value the host wrote before the capture: a comparison, no graph node.
            step_time.set(0.004)
            with self.assertRaisesRegex(RuntimeError, "during a CUDA graph capture"):
                step_time.set(0.002)
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                step_time.fraction(7)
            # A value derived with the setting it was filled with is a comparison; another
            # setting would be a write.
            self.assertIs(step_time.derived("scaled", lambda dt: 3.0 * dt, parameters=3.0), scaled)
            with self.assertRaisesRegex(RuntimeError, "changed during a CUDA graph capture"):
                step_time.derived("scaled", lambda dt: 5.0 * dt, parameters=5.0)
            target.zero_()
            wp.launch(_scaled_by_array, 4, [source, step_time.seconds, target], device=self.device)
        self.assertEqual((step_time.value, step_time.writes), (0.004, 1))
        np.testing.assert_array_equal(scaled.numpy(), [np.float32(3.0 * 0.004)])
        for dt in (0.004, 0.002, 1.0 / 240.0, 0.004):
            step_time.set(dt)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(target.numpy(), np.float32(dt) * values)


class StepKernelTests(unittest.TestCase):
    """Kernels of a step read the time step from an array, never from a launch argument."""

    PACKAGE = "newton_solver_suturebot._src"
    # Parameter names that carry the time step or a value the host derives from it.
    LABELS = ("dt", "h", "timestep", "time_step", "microstep", "scale", "inertia_scale")
    # A scalar time step is a value frozen into every graph that holds the launch. These
    # remain. The mass, spring and triangle blocks of the system matrix are assembled by
    # SolverFEM._build_constant_matrix only, which raises during a capture, so no graph can
    # hold them. The two kernels of the differentiable step belong to a solver whose time
    # step is fixed at construction (differentiation.step rejects any other).
    SCALAR = {
        ("solvers.fem.kernels", "build_system_matrix_diagonal_kernel"),
        ("solvers.fem.kernels", "build_system_matrix_sparse_kernel"),
        ("solvers.fem.kernels", "build_system_matrix_tri_kernel"),
        ("solvers.fem.differentiation", "sum_forces"),
        ("solvers.fem.differentiation", "spring_predictor"),
    }
    # Not a time step of the step being solved, and under another name: the thread's capsule
    # contact relaxes its wrench with a factor that its constructor forms once from the nominal
    # step (ThreadCapsuleContact(dt=...)). Eager and captured steps of every size use that one
    # number. Listed so that this exception is on record.
    CONSTRUCTION = {("solvers.thread.capsules", "_relax"): "relaxation"}
    # The kernels this contract was written for, with the parameters that carry the time step
    # and the type each is read as, which is the type of the former scalar parameter.
    ARRAYS = {
        ("solvers.fem.solver", "accumulate_scaled_kernel"): {"scale": wp.float32},
        ("solvers.fem.kernels", "update_state"): {"dt": wp.float32},
        # The element blocks of the system matrix, which refresh_tangent may issue in a
        # caller's graph (the phantom examples).
        ("solvers.fem.kernels", "build_system_matrix_tet_kernel"): {"dt": wp.float32},
        ("solvers.fem.kernels", "build_system_matrix_hex_kernel"): {"dt": wp.float32},
        ("solvers.fem.implicit_predictor", "restore_spring_damping_predictor"): {"dt": wp.float32},
        ("solvers.fem.prepared_step", "_add_load"): {"dt": wp.float32},
        ("solvers.coupled.interface.insertion", "_surface_forces"): {"dt": wp.float32},
        ("solvers.coupled.interface.insertion", "_shaft_forces"): {"dt": wp.float32},
        ("solvers.coupled.interface.tunnel_alm", "_tunnel_forces"): {"dt": wp.float32},
        ("solvers.coupled.interface.surface_contact_forces", "surface_forces_with_tangent"): {"dt": wp.float32},
        ("solvers.coupled.interface.surface_facet_scratch", "collect_facets"): {"dt": wp.float32},
        ("solvers.coupled.interface.surface_facet_scratch", "forces_from_facets"): {"dt": wp.float32},
        ("solvers.coupled.interface.shaft_projected_kernels", "shaft_forces_projected"): {"dt": wp.float32},
        ("solvers.coupled.interface.shaft_projected_kernels", "tunnel_forces_projected"): {"dt": wp.float32},
        ("solvers.coupled.interface.shaft_projected_kernels", "prepare_anchors_projected"): {"h": wp.float32},
        ("solvers.coupled.interface.shaft_projected_kernels", "prepare_alm_anchors_projected"): {"h": wp.float32},
        ("solvers.coupled.acceleration.shaft_preconditioner", "prepare_anchors"): {"h": wp.float32},
        ("solvers.coupled.acceleration.shaft_preconditioner", "prepare_alm_anchors"): {"h": wp.float32},
        ("solvers.mujoco.rigid_contact_metric", "_prepare_factor"): {"dt": wp.float32},
        ("solvers.mujoco.subcycled_mujoco_warp", "_set_microstep"): {"microstep": wp.float32},
        ("solvers.mujoco.mujoco_free_body_gpu", "integrate"): {"timestep": wp.float64},
        # The thread step: the attachment forms the head's inertia in float32, as before.
        ("solvers.thread.kernels", "predict"): {"dt": wp.float64},
        ("solvers.thread.kernels", "assemble"): {"dt": wp.float64, "inertia_scale": wp.float64},
        ("solvers.thread.kernels", "trial_node_energy"): {"dt": wp.float64},
        ("solvers.thread.kernels", "finish"): {"dt": wp.float64},
        ("solvers.thread.attachment", "_attach"): {"dt": wp.float32},
        ("solvers.thread.attachment", "_start_head"): {"dt": wp.float64},
        ("solvers.thread.thread_rigid_contact", "_find"): {"dt": wp.float64},
        ("solvers.thread.thread_rigid_contact", "_relax"): {"dt": wp.float64},
        # Grips, surface contact and the volume guard of the cutting examples, whose frame
        # is a caller's graph.
        ("cutting.holds", "_add_matrix"): {"scale": wp.float32},
        ("cutting.holds", "_advance_targets"): {"dt": wp.float32},
        ("cutting.holds", "_add_forces"): {"dt": wp.float32},
        ("cutting.contact", "_contact_bounds"): {"dt": wp.float32},
        ("cutting.contact", "_scatter_contacts"): {"dt": wp.float32},
        ("cutting.volume_guard", "_apply"): {"dt": wp.float32},
        ("cutting.volume_guard", "_limit_step"): {"dt": wp.float32},
    }

    def test_no_kernel_of_a_step_takes_the_time_step_as_a_launch_argument(self):
        import importlib
        import pkgutil

        package = importlib.import_module(self.PACKAGE)
        modules, scalar, arrays, construction = 0, set(), {}, {}
        # Every module of the package's source (solvers, cutting, geometry, utilities), not a
        # list that a new module could be missing from.
        for info in pkgutil.walk_packages(package.__path__, self.PACKAGE + "."):
            module = importlib.import_module(info.name)
            name = info.name[len(self.PACKAGE) + 1 :]
            modules += 1
            for kernel_name, kernel in vars(module).items():
                if not isinstance(kernel, wp.Kernel) or kernel.func.__module__ != module.__name__:
                    continue
                for argument in kernel.adj.args:
                    if argument.label == self.CONSTRUCTION.get((name, kernel_name)):
                        construction[(name, kernel_name)] = wp.types.is_array(argument.type)
                    if argument.label not in self.LABELS:
                        continue
                    if wp.types.is_array(argument.type):
                        arrays.setdefault((name, kernel_name), {})[argument.label] = argument.type
                    else:
                        scalar.add((name, kernel_name))
        self.assertGreater(modules, 100)
        self.assertEqual(scalar, self.SCALAR)
        self.assertEqual(construction, dict.fromkeys(self.CONSTRUCTION, False))
        for kernel, parameters in self.ARRAYS.items():
            for parameter, dtype in parameters.items():
                self.assertIn(parameter, arrays.get(kernel, ()), kernel)
                found = arrays[kernel][parameter]
                self.assertEqual((found.dtype, found.ndim), (dtype, 1), (kernel, parameter))


if __name__ == "__main__":
    unittest.main()
