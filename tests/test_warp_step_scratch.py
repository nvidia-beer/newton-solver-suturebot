"""Step scratch: stock MuJoCo Warp steps on a pool of persistent device buffers.

``StepScratch.scope()`` installs a Warp allocator that gives the k-th array created
in the scope the k-th buffer of its pool. ``mujoco_warp.step`` runs inside and
outside such scopes on equal Data of a scene with contacts, joint limits and
position actuators. Requests that reach the built-in allocator of the device are
read from Warp's allocation tracker. It records the native allocation calls, so
memory that passes through no allocator is counted as well.

On the CPU the scope installs nothing and results are compared bit for bit. On
CUDA the stock step does not repeat itself bit for bit: unscoped rollouts of the
scene differ from each other by about 1e-8 in qvel. A scoped rollout is
therefore compared with a stock one to 1e-3 of the largest value, the tolerance
of the other MuJoCo Warp rollout comparisons in these tests. With deterministic
atomics the stock step does repeat itself on CUDA, and scoped steps are compared
with it bit for bit.

    PYTHONPATH=tests python -m unittest test_warp_step_scratch -v
    NEWTON_TEST_DEVICE=cuda:0 PYTHONPATH=tests python -m unittest test_warp_step_scratch -v
"""

import threading
import unittest

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from _warp_step_support import ARM, CUDA, DEVICE, DeviceTestCase, allocation_tracker, allocations, count_down

from newton_solver_suturebot._src.solvers.mujoco.warp_step.scratch import StepScratch

STEPS = 20


def _model(deterministic=False):
    """Device model and data of the arm scene in a moving contact state; equal for every call."""
    mjm = mujoco.MjModel.from_xml_string(ARM)
    mjd = mujoco.MjData(mjm)
    mjd.ctrl[:] = (0.9, 1.5, 0.08)  # targets past the joint ranges
    mujoco.mj_step(mjm, mjd, 60)
    mjd.qvel[:] += np.random.default_rng(5).uniform(-0.3, 0.3, mjm.nv)
    mujoco.mj_step(mjm, mjd, 5)
    mujoco.mj_forward(mjm, mjd)
    mjd.qacc_warmstart[:] = mjd.qacc
    m = mjw.put_model(mjm)
    m.opt.deterministic = deterministic
    return m, mjw.put_data(mjm, mjd)


def _state(d):
    return {"qpos": d.qpos.numpy(), "qvel": d.qvel.numpy()}


def _rollout(steps, scratch=None, deterministic=False):
    """State after the steps from the scene's initial state, and the solver iterations of every step."""
    m, d = _model(deterministic)
    iterations = []
    for _ in range(steps):
        if scratch is None:
            mjw.step(m, d)
        else:
            with scratch.scope():
                mjw.step(m, d)
        iterations.append(int(d.solver_niter.numpy().max()))
    return _state(d), iterations


def _requests(scratch, *sizes):
    """Pointers that one scope hands to byte arrays of the given sizes; every array dies at once."""
    with scratch.scope():
        return [wp.empty(size, dtype=wp.uint8).ptr for size in sizes]


class StepScratchTests(DeviceTestCase):
    def _assert_same(self, actual, expected):
        for name, value in expected.items():
            self.assertTrue(np.isfinite(value).all(), name)
            if CUDA:
                bound = 1e-3 * float(np.abs(value).max())
                np.testing.assert_allclose(actual[name], value, rtol=0.0, atol=bound, err_msg=name)
            else:
                np.testing.assert_array_equal(actual[name], value, err_msg=name)

    def test_scoped_steps_match_stock_steps(self):
        expected, iterations = _rollout(STEPS)
        scratch = StepScratch(DEVICE)
        actual, _ = _rollout(STEPS, scratch)
        self.assertGreater(max(iterations), 1)
        self._assert_same(actual, expected)
        # the CPU scope leaves the pool empty
        self.assertEqual(scratch.buffer_count > 0, CUDA)

    def test_cpu_scope_installs_nothing(self):
        scratch = StepScratch("cpu")
        self.assertIsInstance(scratch, wp.Allocator)
        signature = scratch.signature
        allocator = wp.get_device_allocator("cpu")
        with scratch.scope():
            self.assertIs(wp.get_device_allocator("cpu"), allocator)
            values = wp.full(5, 2.0, dtype=float, device="cpu")
            with self.assertRaisesRegex(RuntimeError, "do not nest"):
                with scratch.scope():
                    pass
        with scratch.scope():
            other = wp.full(5, 3.0, dtype=float, device="cpu")
        np.testing.assert_array_equal(values.numpy(), 2.0)
        np.testing.assert_array_equal(other.numpy(), 3.0)
        self.assertEqual(scratch.buffer_count, 0)
        self.assertEqual(scratch.signature, signature)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_second_scoped_step_requests_no_device_memory(self):
        m, d = _model()
        self.assertEqual(m.nu, 3)
        scratch = StepScratch(DEVICE)
        requested, iterations, signatures = [], [], []
        with allocation_tracker() as tracker:
            for _ in range(4):
                before = allocations(tracker)
                with scratch.scope():
                    mjw.step(m, d)
                requested.append(allocations(tracker) - before)
                iterations.append(int(d.solver_niter.numpy()[0]))
                signatures.append(scratch.signature)
            before = allocations(tracker)
            mjw.step(m, d)
            stock = allocations(tracker) - before
        self.assertGreater(int(d.nacon.numpy()[0]), 0)
        self.assertGreater(int(d.nl.numpy()[0]), 0)
        # the solver loop ran a different number of times in these steps
        self.assertGreater(len(set(iterations)), 1)
        self.assertGreater(scratch.buffer_count, 0)
        self.assertEqual(requested, [scratch.buffer_count, 0, 0, 0])
        self.assertEqual(len(set(signatures)), 1)
        # every device allocation of a stock step is an array, which the pool serves
        self.assertEqual(stock, scratch.buffer_count)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_scoped_steps_equal_stock_steps_with_deterministic_atomics(self):
        expected, iterations = _rollout(STEPS, deterministic=True)
        repeated, _ = _rollout(STEPS, deterministic=True)
        for name, value in expected.items():
            np.testing.assert_array_equal(repeated[name], value, err_msg=f"stock {name}")
        scratch = StepScratch(DEVICE)
        m, d = _model(deterministic=True)
        signatures = set()
        for _ in range(STEPS):
            with scratch.scope():
                mjw.step(m, d)
            signatures.add(scratch.signature)
        self.assertGreater(max(iterations), 1)
        # Warp's deterministic launches create buffers whose number and sizes follow the
        # solver iterations and the data, so buffers were replaced between these scopes.
        self.assertGreater(len(signatures), 1)
        for name, value in expected.items():
            self.assertTrue(np.isfinite(value).all(), name)
            np.testing.assert_array_equal(_state(d)[name], value, err_msg=name)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_scope_hands_out_the_same_distinct_buffers(self):
        scratch = StepScratch(DEVICE)
        source = wp.array(np.arange(24, dtype=np.float32).reshape(4, 6), dtype=float)
        host = np.arange(5, dtype=np.int32)

        def create():
            return [
                wp.zeros(7, dtype=wp.vec3),
                wp.empty((3, 5), dtype=int),
                wp.empty_like(source),
                wp.zeros_like(source),
                wp.full(9, 2.5, dtype=float),
                wp.clone(source),
                wp.array(host, dtype=int),
            ]

        sizes = [84, 60, 96, 96, 36, 96, 20]
        with allocation_tracker() as tracker:
            before = allocations(tracker)
            with scratch.scope():
                self.assertIs(wp.get_device_allocator(DEVICE), scratch)
                arrays = create()
            first = allocations(tracker) - before
            self.assertEqual([array.capacity for array in arrays], sizes)
            pointers = [array.ptr for array in arrays]
            spans = sorted(zip(pointers, sizes, strict=True))
            for (low, size), (high, _) in zip(spans, spans[1:], strict=False):
                self.assertLessEqual(low + size, high)
            # leave values behind that the next scope must not show in initialized arrays
            for array in arrays:
                array.fill_(7)
            del arrays, array
            signature = scratch.signature
            before = allocations(tracker)
            with scratch.scope():
                arrays = create()
            second = allocations(tracker) - before
        # one request per array reached the built-in allocator, and none in the second scope
        self.assertEqual(first, len(sizes))
        self.assertEqual(second, 0)
        self.assertEqual(scratch.buffer_count, len(sizes))
        self.assertEqual(scratch.signature, signature)
        self.assertEqual([array.ptr for array in arrays], pointers)
        zeros, _, _, zeros_like, full, clone, data = arrays
        np.testing.assert_array_equal(zeros.numpy(), 0.0)
        np.testing.assert_array_equal(zeros_like.numpy(), 0.0)
        np.testing.assert_array_equal(full.numpy(), 2.5)
        np.testing.assert_array_equal(clone.numpy(), source.numpy())
        np.testing.assert_array_equal(data.numpy(), host)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_captured_step_requests_no_device_memory(self):
        expected, _ = _rollout(STEPS)
        scratch = StepScratch(DEVICE)
        m, d = _model()
        with scratch.scope():
            mjw.step(m, d)
        signature = scratch.signature
        m, d = _model()
        with allocation_tracker() as tracker:
            before = allocations(tracker)
            with wp.ScopedCapture() as capture:
                with scratch.scope():
                    mjw.step(m, d)
            captured = allocations(tracker) - before
            for _ in range(STEPS):
                wp.capture_launch(capture.graph)
            replayed = allocations(tracker) - before
            # the same capture without the scope allocates every temporary array of the step
            stock_m, stock_d = _model()
            before = allocations(tracker)
            with wp.ScopedCapture():
                mjw.step(stock_m, stock_d)
            stock = allocations(tracker) - before
        self.assertEqual(captured, 0)
        self.assertEqual(replayed, 0)
        self.assertEqual(stock, scratch.buffer_count)
        self.assertEqual(scratch.signature, signature)
        self._assert_same(_state(d), expected)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_steps_in_conditional_body_request_no_device_memory(self):
        expected, _ = _rollout(STEPS)
        scratch = StepScratch(DEVICE)
        m, d = _model()
        with scratch.scope():
            mjw.step(m, d)
        signature = scratch.signature
        m, d = _model()
        remaining = wp.array([STEPS // 4], dtype=int)

        def microsteps():
            for _ in range(4):
                with scratch.scope():
                    mjw.step(m, d)
            wp.launch(count_down, dim=1, inputs=[remaining])

        with allocation_tracker() as tracker:
            before = allocations(tracker)
            with wp.ScopedCapture() as capture:
                wp.capture_while(remaining, microsteps)
            wp.capture_launch(capture.graph)
            requested = allocations(tracker) - before
        self.assertEqual(requested, 0)
        self.assertEqual(scratch.signature, signature)
        self.assertEqual(int(remaining.numpy()[0]), 0)
        self._assert_same(_state(d), expected)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_stock_step_in_conditional_body_is_refused(self):
        # Warp does not capture a conditional body that allocates memory, which is what
        # every temporary array of a stock step does without the scope.
        m, d = _model()
        remaining = wp.array([1], dtype=int)

        def microstep():
            mjw.step(m, d)
            wp.launch(count_down, dim=1, inputs=[remaining])

        with self.assertRaisesRegex(RuntimeError, "memory allocation"):
            with wp.ScopedCapture():
                wp.capture_while(remaining, microstep)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_request_outside_the_pool_raises_during_capture(self):
        scratch = StepScratch(DEVICE)
        with self.assertRaisesRegex(RuntimeError, "request 0 for 16 bytes during graph capture.*no buffer"):
            with wp.ScopedCapture():
                _requests(scratch, 16)
        self.assertEqual(scratch.buffer_count, 0)
        pointers = _requests(scratch, 16, 32)
        signature = scratch.signature
        with self.assertRaisesRegex(RuntimeError, "request 1 for 64 bytes during graph capture.*buffer of 32 bytes"):
            with wp.ScopedCapture():
                _requests(scratch, 16, 64)
        with self.assertRaisesRegex(RuntimeError, "request 2 for 8 bytes during graph capture.*no buffer"):
            with wp.ScopedCapture():
                _requests(scratch, 16, 32, 8)
        self.assertEqual(scratch.buffer_count, 2)
        self.assertEqual(scratch.signature, signature)
        # The sequence of the pool is captured, here with the scope opened before the capture,
        # and the graph writes the buffers of the pool.
        with scratch.scope():
            with wp.ScopedCapture() as capture:
                wp.empty(16, dtype=wp.uint8)
                wp.full(8, 3.0, dtype=float)
        self.assertEqual(scratch.signature, signature)
        with scratch.scope():
            wp.empty(16, dtype=wp.uint8)
            wp.full(8, 7.0, dtype=float)
        wp.capture_launch(capture.graph)
        with scratch.scope():
            wp.empty(16, dtype=wp.uint8)
            left = wp.empty(8, dtype=float)
            self.assertEqual(left.ptr, pointers[1])
            np.testing.assert_array_equal(left.numpy(), 3.0)
        self.assertEqual(scratch.buffer_count, 2)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_pool_replaces_buffers_outside_capture(self):
        scratch = StepScratch(DEVICE)
        with allocation_tracker() as tracker:
            before, live = allocations(tracker), allocations(tracker, "Live")
            pointers = _requests(scratch, 16, 32)
            signature = scratch.signature
            self.assertEqual(_requests(scratch, 16, 32), pointers)
            self.assertEqual(scratch.signature, signature)
            self.assertEqual(allocations(tracker) - before, 2)
            # other sizes at both positions
            swapped = _requests(scratch, 32, 16)
            self.assertNotEqual(scratch.signature, signature)
            self.assertEqual(scratch.buffer_count, 2)
            self.assertNotIn(swapped[0], pointers)
            # the buffers that left their positions stay allocated and are taken into use again:
            # alternating between the two sequences stops asking the device for memory
            _requests(scratch, 16, 32)
            self.assertEqual(allocations(tracker) - before, 4)
            self.assertEqual(allocations(tracker, "Live") - live, 4)
            signatures = {signature}
            for _ in range(3):
                _requests(scratch, 32, 16)
                signatures.add(scratch.signature)
                _requests(scratch, 16, 32)
                signatures.add(scratch.signature)
            self.assertEqual(len(signatures), 7)
            self.assertEqual(allocations(tracker) - before, 4)
            # a longer sequence adds buffers
            self.assertEqual(_requests(scratch, 16, 32, 8)[:2], _requests(scratch, 16, 32))
            self.assertEqual(scratch.buffer_count, 3)
            self.assertEqual(allocations(tracker) - before, 5)
            scratch.close()
            self.assertEqual(allocations(tracker, "Live") - live, 0)
        self.assertEqual(scratch.buffer_count, 0)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_array_outliving_its_scope_keeps_its_buffer(self):
        scratch = StepScratch(DEVICE)
        with scratch.scope():
            kept = wp.full(8, 1.5, dtype=float)
        signature = scratch.signature
        with self.assertRaisesRegex(RuntimeError, "request 0 for 32 bytes during graph capture.*still holds"):
            with wp.ScopedCapture():
                with scratch.scope():
                    wp.zeros(8, dtype=float)
        self.assertEqual(scratch.signature, signature)
        with scratch.scope():
            other = wp.full(8, 2.5, dtype=float)
        self.assertNotEqual(other.ptr, kept.ptr)
        self.assertNotEqual(scratch.signature, signature)
        self.assertEqual(scratch.buffer_count, 1)
        np.testing.assert_array_equal(kept.numpy(), 1.5)
        np.testing.assert_array_equal(other.numpy(), 2.5)
        # once the arrays are gone the position has one buffer for every scope again
        pointer = other.ptr
        del kept, other
        signature = scratch.signature
        self.assertEqual(_requests(scratch, 32), [pointer])
        self.assertEqual(scratch.signature, signature)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_scope_ends_cleanly(self):
        scratch = StepScratch(DEVICE)
        allocator = wp.get_device_allocator(DEVICE)
        with self.assertRaisesRegex(RuntimeError, "only inside scope"):
            scratch.allocate(16)
        with scratch.scope():
            first = wp.zeros(4, dtype=float).ptr
            with self.assertRaisesRegex(RuntimeError, "do not nest"):
                with scratch.scope():
                    pass
            self.assertIs(wp.get_device_allocator(DEVICE), scratch)
            second = wp.zeros(4, dtype=float).ptr
        self.assertNotEqual(first, second)
        self.assertIs(wp.get_device_allocator(DEVICE), allocator)
        with self.assertRaises(ZeroDivisionError):
            with scratch.scope():
                wp.zeros(4, dtype=float)
                raise ZeroDivisionError
        self.assertIs(wp.get_device_allocator(DEVICE), allocator)
        self.assertEqual(_requests(scratch, 16, 16), [first, second])
        # arrays created after the scope come from the built-in allocator
        outside = wp.zeros(4, dtype=float)
        self.assertNotIn(outside.ptr, (first, second))

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_other_thread_is_refused(self):
        scratch = StepScratch(DEVICE)
        errors = []

        def create():
            try:
                wp.zeros(4, dtype=float, device=DEVICE)
            except RuntimeError as error:
                errors.append(str(error))

        with scratch.scope():
            wp.zeros(4, dtype=float)
            thread = threading.Thread(target=create)
            thread.start()
            thread.join()
            wp.zeros(4, dtype=float)
        self.assertEqual(len(errors), 1)
        self.assertIn("another thread", errors[0])
        # the refused request took no position
        self.assertEqual(scratch.buffer_count, 2)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_close_releases_buffers(self):
        scratch = StepScratch(DEVICE)
        with allocation_tracker() as tracker:
            live = allocations(tracker, "Live")
            with scratch.scope():
                kept = wp.full(8, 4.0, dtype=float)
                wp.zeros(16, dtype=float)
                with self.assertRaisesRegex(RuntimeError, "inside scope"):
                    scratch.close()
            signature = scratch.signature
            self.assertEqual(allocations(tracker, "Live") - live, 2)
            scratch.close()
            self.assertEqual(scratch.buffer_count, 0)
            self.assertNotEqual(scratch.signature, signature)
            # the buffer of the living array is released with that array
            self.assertEqual(allocations(tracker, "Live") - live, 1)
            np.testing.assert_array_equal(kept.numpy(), 4.0)
            del kept
            self.assertEqual(allocations(tracker, "Live") - live, 0)
            # the pool grows again
            self.assertEqual(len(_requests(scratch, 16)), 1)
            self.assertEqual(scratch.buffer_count, 1)
            self.assertEqual(allocations(tracker, "Live") - live, 1)


if __name__ == "__main__":
    unittest.main()
