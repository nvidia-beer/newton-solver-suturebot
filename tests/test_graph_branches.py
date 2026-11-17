"""Concurrent branches of one captured CUDA graph: results and dependency edits."""

import os
import unittest

import warp as wp

from newton_solver_suturebot._src.utils.graph_branches import Fork

ITERATIONS, KERNELS, INNER, LOOPS = 6, 12, 3, 2


@wp.kernel(enable_backward=False)
def _bump(values: wp.array[int]):
    values[0] = values[0] + 1


@wp.kernel(enable_backward=False)
def _countdown(counter: wp.array[int], running: wp.array[int]):
    counter[0] = counter[0] - 1
    if counter[0] <= 0:
        running[0] = 0


@wp.kernel(enable_backward=False)
def _reset(counter: wp.array[int], running: wp.array[int], count: int):
    counter[0] = count
    running[0] = 1


@wp.kernel(enable_backward=False)
def _record_sum(first: wp.array[int], second: wp.array[int], total: wp.array[int]):
    # After a join both branches of the same iteration are complete.
    total[0] = total[0] + first[0] + second[0]


class GraphBranchTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def test_no_fork_outside_a_cuda_capture(self):
        self.assertIsNone(Fork.begin("cpu"))
        if self.device.is_cuda:
            self.assertIsNone(Fork.begin(self.device))

    def _arrays(self, count):
        return [wp.zeros(1, dtype=int, device=self.device) for _ in range(count)]

    def _loops(self, values, counter, running, after_conditional=None):
        """Several data-dependent loops in sequence, each a conditional graph node."""
        for _ in range(LOOPS):

            def body():
                for _ in range(KERNELS // INNER):
                    wp.launch(_bump, 1, [values], device=self.device)
                wp.launch(_countdown, 1, [counter, running], device=self.device)

            wp.launch(_reset, 1, [counter, running, INNER], device=self.device)
            wp.capture_while(running, body)
            if after_conditional is not None:
                after_conditional()

    def test_branches_with_conditionals_run_every_iteration_and_join(self):
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Graph branches need CUDA conditional graphs")
        first, second, total, outer, running, c1, r1, c2, r2 = self._arrays(9)
        for kernel, arguments in (
            (_bump, [first]),
            (_countdown, [outer, running]),
            (_reset, [outer, running, 1]),
            (_record_sum, [first, second, total]),
        ):
            wp.launch(kernel, 1, arguments, device=self.device)
        observed = {}

        def serial():
            self._loops(first, c1, r1)
            self._loops(second, c2, r2)
            wp.launch(_record_sum, 1, [first, second, total], device=self.device)

        def forked():
            fork = Fork.begin(self.device)
            self.assertIsNotNone(fork)
            start = list(fork._start)
            with fork.branch():
                self._loops(first, c1, r1)
            ends = list(fork._ends)
            with fork.branch():
                # The second branch starts again from the nodes before the fork.
                observed["restart"] = fork._capture()[1] == start
                self._loops(second, c2, r2, fork.isolate)
                # After a conditional and isolate() it no longer waits for the first branch.
                observed["isolated"] = not set(fork._capture()[1]) & set(ends)
            fork.join()
            observed["joined"] = set(fork._capture()[1]) == set(fork._ends) and set(ends) < set(fork._ends)
            wp.launch(_record_sum, 1, [first, second, total], device=self.device)
            with self.assertRaises(RuntimeError):
                fork.join()
            with self.assertRaises(RuntimeError):
                fork.isolate()

        results = []
        for body in (serial, forked):

            def loop(body=body):
                body()
                wp.launch(_countdown, 1, [outer, running], device=self.device)

            with wp.ScopedCapture(device=self.device) as capture:
                wp.launch(_reset, 1, [outer, running, ITERATIONS], device=self.device)
                wp.capture_while(running, loop)
            for _ in range(2):
                for array in (first, second, total):
                    array.zero_()
                wp.capture_launch(capture.graph)
                results.append([int(array.numpy()[0]) for array in (first, second, total)])
        per_branch = ITERATIONS * LOOPS * KERNELS
        # Sum over iterations of both running totals, read after each join.
        after_joins = 2 * LOOPS * KERNELS * ITERATIONS * (ITERATIONS + 1) // 2
        self.assertEqual(results, [[per_branch, per_branch, after_joins]] * 4)
        self.assertEqual(observed, {"restart": True, "isolated": True, "joined": True})

    def test_branches_cannot_be_nested(self):
        if not self.device.is_cuda:
            self.skipTest("Graph branches need CUDA")
        values = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(_bump, 1, [values], device=self.device)
        with wp.ScopedCapture(device=self.device):
            wp.launch(_bump, 1, [values], device=self.device)
            fork = Fork.begin(self.device)
            self.assertIsNotNone(fork)
            with fork.branch():
                wp.launch(_bump, 1, [values], device=self.device)
                with self.assertRaises(RuntimeError):
                    with fork.branch():
                        pass
            fork.join()
            with self.assertRaises(RuntimeError):
                with fork.branch():
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
