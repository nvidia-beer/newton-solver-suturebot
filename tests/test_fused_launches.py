"""Fused launches: a recorded run of launches and memory operations as stages of generated kernels.

``FusedLaunches`` must leave in the arrays what issuing the recorded operations one after
the other leaves there. The synthetic tests issue a sequence both ways from equal arrays
and compare the arrays bit for bit: every kernel computes integers, or sums whose order
does not change the result, so the order of the threads of a launch cannot show. The
sequences are written once (``_Sequence``) as calls of Warp and as the records that stand
for them. On CUDA ``record`` must return exactly those records; on the CPU, where
``record`` has no graph to read, the records are used as written. There one thread runs
the stages of a generated kernel in their order, so a wrong index or a wrong order of
stages shows without any race; a missing block barrier can only show on CUDA.

The MuJoCo Warp tests record stages of ``mujoco_warp.step`` on the scenes of the rigid
step tests and compare every array of Data with the stock call from equal Data: bit for
bit where two stock calls agree bit for bit, else to 1e-6 of the largest value.

    PYTHONPATH=tests python -m unittest test_fused_launches -v
    NEWTON_TEST_DEVICE=cuda:0 PYTHONPATH=tests python -m unittest test_fused_launches -v
"""

import gc
import os
import re
import subprocess
import sys
import unittest
import weakref
from typing import Any
from unittest import mock

import mujoco_warp as mjw
import numpy as np
import warp as wp
from _warp_step_support import (
    CUDA,
    DEVICE,
    DeviceTestCase,
    allocation_tracker,
    allocations,
    clone_data,
    count_down,
    data_arrays,
    step_model,
)
from mujoco_warp._src import collision_driver, constraint, forward, passive, smooth

from newton_solver_suturebot._src.solvers.mujoco.warp_step.scratch import StepScratch
from newton_solver_suturebot._src.utils import fused_launches
from newton_solver_suturebot._src.utils.fused_launches import FusedLaunches, Launch, Memcpy, Memset, record
from newton_solver_suturebot._src.utils.graph_nodes import graph_nodes

CUDA_ONLY = "record() and graph capture need a CUDA device (NEWTON_TEST_DEVICE=cuda:0)"
TILE = 32
# Threads per block of the launches that the tape of a device records without an explicit block.
DEFAULT_BLOCK = 256 if CUDA else 1


@wp.kernel(enable_backward=False)
def _ramp(out: wp.array[int], scale: int):
    i = wp.tid()
    out[i] = scale * i + 1


@wp.kernel(enable_backward=False)
def _accumulate(first: wp.array[int], second: wp.array[int], total: wp.array[int]):
    i = wp.tid()
    total[i] = 3 * total[i] + first[i] + 2 * second[i]


@wp.kernel(enable_backward=False)
def _grid(out: wp.array2d[int]):
    i, j = wp.tid()
    out[i, j] = 100 * i + j


@wp.kernel(enable_backward=False)
def _volume(grid: wp.array2d[int], out: wp.array3d[int]):
    i, j, k = wp.tid()
    out[i, j, k] = 10000 * i + 100 * j + k + grid[i, j]


@wp.kernel(enable_backward=False)
def _leading(count: wp.array[int]):
    """One index name for a launch of two dimensions: the threads of a row share its index."""
    i = wp.tid()
    wp.atomic_add(count, i, i + 1)


@wp.kernel(enable_backward=False)
def _padded(out: wp.array2d[int]):
    """Two index names for a launch of one dimension: the second index is zero."""
    i, j = wp.tid()
    out[i, j] = out[i, j] + 7 + i + 1000 * j


@wp.kernel(enable_backward=False)
def _count(counter: wp.array[int]):
    """No thread index: every thread of the launch does the same."""
    wp.atomic_add(counter, 0, 1)


@wp.kernel(enable_backward=False)
def _slow_write(values: wp.array[int]):
    """Threads with a higher index write later, so a reader that does not wait sees stale values."""
    i = wp.tid()
    total = int(0)
    for k in range(i * 300):
        total += (k + values[i]) % 7
    values[i] = total + i


@wp.kernel(enable_backward=False)
def _mirror(values: wp.array[int], out: wp.array[int]):
    """Reads what other threads of the stage before wrote."""
    i = wp.tid()
    count = values.shape[0]
    out[i] = values[count - 1 - i] + 2 * values[(i + 1) % count]


@wp.kernel(enable_backward=False)
def _first_match(table: wp.array2d[int], out: wp.array[int]):
    """Returns from inside two nested loops and from the end of the body."""
    i = wp.tid()
    out[i] = -1
    if table[i, 0] < 0:
        return
    for j in range(table.shape[1]):
        for k in range(j + 1):
            if table[i, j] == 10 * j + k:
                out[i] = 100 * j + k
                return
    out[i] = -2


@wp.kernel(enable_backward=False)
def _histogram(keys: wp.array2d[int], bins: wp.array[int]):
    i, j = wp.tid()
    wp.atomic_add(bins, keys[i, j], 1)


@wp.kernel(enable_backward=False)
def _append(values: wp.array[int], count: wp.array[int], slots: wp.array[int]):
    """Takes slots from a shared counter; the order of the slots is that of the threads."""
    i = wp.tid()
    if values[i] % 3 == 0:
        slot = wp.atomic_add(count, 0, 1)
        slots[slot] = values[i]


@wp.kernel(enable_backward=False)
def _add_bytes(source: wp.array[wp.uint8], total: wp.array[int]):
    i = wp.tid()
    total[i] = 5 * total[i] + int(source[i])


@wp.kernel(enable_backward=False)
def _add_floats(source: wp.array[float], total: wp.array[int]):
    i = wp.tid()
    total[i] = 7 * total[i] + int(4.0 * source[i])


@wp.kernel(enable_backward=False)
def _add_longs(source: wp.array[wp.int64], total: wp.array[int]):
    i = wp.tid()
    total[i] = 11 * total[i] + int(source[i] % wp.int64(1000))


@wp.kernel(enable_backward=False)
def _add_shorts(source: wp.array[wp.int16], total: wp.array[int]):
    i = wp.tid()
    total[i] = 13 * total[i] + int(source[i])


@wp.kernel(enable_backward=False)
def _expression_index(out: wp.array[int]):
    """The thread index inside an expression."""
    out[wp.tid()] = out[wp.tid()] + 5


@wp.kernel(enable_backward=False)
def _nested_index(out: wp.array[int]):
    """The thread index assigned inside a block."""
    if out.shape[0] > 0:
        i = wp.tid()
        out[i] = out[i] + 9


@wp.kernel(enable_backward=False)
def _generic_add(out: wp.array[Any], increment: Any):
    i = wp.tid()
    out[i] = out[i] + increment


@wp.kernel(enable_backward=False, cuda_max_registers=64)
def _register_limited(out: wp.array[int]):
    i = wp.tid()
    out[i] = out[i] + 1


@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def _fast_math_add(out: wp.array[int]):
    i = wp.tid()
    out[i] = out[i] + 2 * i


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN}, enable_backward=False)
def _deterministic_add(out: wp.array[int]):
    i = wp.tid()
    wp.atomic_add(out, i, 1)


@wp.kernel(enable_backward=False)
def _tile_sum(values: wp.array2d[float], out: wp.array[float]):
    """A tile kernel of Warp: the threads of a block load one row and add it up together."""
    i = wp.tid()
    tile = wp.tile_load(values[i], shape=TILE)
    wp.tile_store(out, wp.tile_sum(tile), offset=i)


@wp.func_native(snippet="WP_TILE_SYNC();")
def _barrier(): ...


@wp.kernel(enable_backward=False)
def _block_reverse(values: wp.array2d[int], scratch: wp.array2d[int]):
    """A kernel that strides over a row with the threads of its block, as the kernels of the rigid step do."""
    row, thread = wp.tid()
    count = values.shape[1]
    for i in range(thread, count, wp.block_dim()):
        scratch[row, i] = values[row, i] + 1
    _barrier()
    for i in range(thread, count, wp.block_dim()):
        values[row, i] = scratch[row, count - 1 - i]


@wp.func
def _synchronized_increment(value: int):
    _barrier()
    return value + 1


@wp.kernel(enable_backward=False)
def _calls_native(out: wp.array[int]):
    """A plain kernel that reaches native code through a Warp function."""
    i = wp.tid()
    out[i] = _synchronized_increment(out[i])


@wp.kernel(enable_backward=False)
def _world_ramp(out: wp.array2d[int], scale: int):
    worldid, i = wp.tid()
    out[worldid, i] = 1000 * worldid + scale * i


@wp.kernel(enable_backward=False)
def _world_mirror(values: wp.array2d[int], out: wp.array2d[int]):
    worldid, i = wp.tid()
    count = values.shape[1]
    out[worldid, i] = values[worldid, count - 1 - i] + 2 * values[worldid, (i + 1) % count] + out[worldid, i]


@wp.kernel(enable_backward=False)
def _world_total(values: wp.array2d[int], total: wp.array[int]):
    worldid = wp.tid()
    result = int(0)
    for i in range(values.shape[1]):
        result += values[worldid, i]
    total[worldid] = 3 * total[worldid] + result


@wp.kernel(enable_backward=False)
def _next_world(values: wp.array2d[int], out: wp.array2d[int]):
    """Led by an index that is not the world index: reads the world after its own."""
    w, i = wp.tid()
    out[w, i] = out[w, i] + values[(w + 1) % values.shape[0], i]


@wp.kernel(enable_backward=False)
def _pairs(values: wp.array2d[int], out: wp.array2d[int]):
    """A launch over all items of all worlds that reads the world before its own."""
    item = wp.tid()
    worlds, count = values.shape[0], values.shape[1]
    w, i = item // count, item % count
    out[w, i] = 2 * out[w, i] + values[(w + worlds - 1) % worlds, i]


def _scaled(scale, offset):
    """A kernel with constants of its closure, one of them through a static expression."""

    @wp.kernel(name="scaled", module="unique", enable_backward=False)
    def kernel(out: wp.array[int]):
        i = wp.tid()
        out[i] = out[i] * scale + wp.static(offset * 3) + i

    return kernel


def _scaled_twice(scale, offset):
    """A second factory: the name and the source of the kernels of ``_scaled``, twice its scale."""
    scale = 2 * scale

    @wp.kernel(name="scaled", module="unique", enable_backward=False)
    def kernel(out: wp.array[int]):
        i = wp.tid()
        out[i] = out[i] * scale + wp.static(offset * 3) + i

    return kernel


def _bits(array, value):
    """One element of an array with ``value`` as the unsigned integer and the bytes that a fill writes."""
    element = np.asarray(value, dtype=wp.dtype_to_numpy(array.dtype)).tobytes()
    return int.from_bytes(element, "little"), len(element)


class _Sequence:
    """Launches and memory operations as calls of Warp and as the records that stand for them."""

    def __init__(self):
        self.operations = []
        self._calls = []

    def launch(self, kernel, dim, inputs=(), outputs=(), block_dim=0, max_blocks=0):
        dim = (dim,) if isinstance(dim, int) else tuple(dim)
        inputs, outputs = tuple(inputs), tuple(outputs)
        self._calls.append(
            lambda: wp.launch(
                kernel, dim=dim, inputs=inputs, outputs=outputs, block_dim=block_dim, max_blocks=max_blocks
            )
        )
        block = (block_dim or DEFAULT_BLOCK) if CUDA else 1
        self.operations.append(Launch(kernel, dim, inputs, outputs, block, max_blocks))

    def launch_tiled(self, kernel, dim, inputs, block_dim):
        inputs = tuple(inputs)
        self._calls.append(lambda: wp.launch_tiled(kernel, dim=list(dim), inputs=inputs, block_dim=block_dim))
        block = block_dim if CUDA else 1
        self.operations.append(Launch(kernel, (*dim, block), inputs, (), block))

    def zero(self, array):
        self._calls.append(array.zero_)
        self.operations.append(Memset(array.ptr, 0, 1, array.size * wp.types.type_size_in_bytes(array.dtype), (array,)))

    def fill(self, array, value):
        self._calls.append(lambda: array.fill_(value))
        bits, width = _bits(array, value)
        self.operations.append(Memset(array.ptr, bits, width, array.size, (array,)))

    def copy(self, dest, src, dest_offset=0, src_offset=0, count=None):
        count = src.size if count is None else count
        self._calls.append(lambda: wp.copy(dest, src, dest_offset=dest_offset, src_offset=src_offset, count=count))
        item = wp.types.type_size_in_bytes(src.dtype)
        self.operations.append(
            Memcpy(dest.ptr + dest_offset * item, src.ptr + src_offset * item, count * item, (dest, src))
        )

    def __call__(self):
        for call in self._calls:
            call()


def _signature(operations):
    """What a list of records says, without the arrays that keep memory alive."""
    lines = []
    for operation in operations:
        if isinstance(operation, Launch):
            values = tuple(
                value.ptr if isinstance(value, wp.array) else value for value in (*operation.inputs, *operation.outputs)
            )
            lines.append(("launch", operation.kernel.key, operation.dim, values, operation.block_dim))
        elif isinstance(operation, Memset):
            lines.append(("memset", operation.pointer, operation.value, operation.width, operation.count))
        else:
            lines.append(("memcpy", operation.destination, operation.source, operation.size))
    return lines


def _kernel_name(fused, node=0):
    """Name of the generated kernel that a node of ``fused`` launches."""
    return fused.describe()[node].split(":")[0]


def _arguments(fused, node=0):
    """Number of arguments of the generated kernel that a node of ``fused`` launches."""
    return int(re.search(r"(\d+) arguments", fused.describe()[node]).group(1))


def _ints(count, seed=0):
    return wp.array(np.random.default_rng(seed).integers(0, 50, count), dtype=int)


class _SequenceTestCase(DeviceTestCase):
    def _recorded(self, sequence):
        """The records of a sequence: read by ``record`` on CUDA, which must agree with the sequence."""
        if not CUDA:
            return sequence.operations
        operations = record(sequence, DEVICE)
        self.assertEqual(_signature(operations), _signature(sequence.operations))
        return operations

    def _assert_arrays(self, arrays, expected):
        for index, (array, value) in enumerate(zip(arrays, expected, strict=True)):
            np.testing.assert_array_equal(array.numpy(), value, err_msg=f"array {index}")

    def _check(self, sequence, arrays, nworld=1, nodes=None):
        """Fused launches of a sequence against its calls, eagerly and replayed from a graph."""
        start = [array.numpy().copy() for array in arrays]

        def restart():
            for array, value in zip(arrays, start, strict=True):
                array.assign(value)

        sequence()
        expected = [array.numpy().copy() for array in arrays]
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(start, expected, strict=True)))
        restart()
        operations = self._recorded(sequence)
        self._assert_arrays(arrays, start)  # recording runs nothing
        fused = FusedLaunches(operations, nworld, DEVICE)
        if nodes is not None:
            self.assertEqual(fused.node_count, nodes, "\n".join(fused.describe()))
        self.assertEqual(len(fused.describe()), fused.node_count)
        fused.launch()
        self._assert_arrays(arrays, expected)
        if CUDA:
            restart()
            with wp.ScopedCapture() as capture:
                fused.launch()
            self.assertEqual(len(graph_nodes(capture.graph)), fused.node_count)
            self._assert_arrays(arrays, start)
            wp.capture_launch(capture.graph)
            self._assert_arrays(arrays, expected)
        restart()
        return fused


class StageTests(_SequenceTestCase):
    def test_launch_dimensions(self):
        line, grid, volume = wp.zeros(37, dtype=int), wp.zeros((5, 7), dtype=int), wp.zeros((3, 5, 7), dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 37, [line, 3])
        sequence.launch(_grid, (5, 7), [grid])
        sequence.launch(_volume, (3, 5, 7), [grid, volume])
        sequence.launch(_ramp, (37,), [line, 4])
        fused = self._check(sequence, [line, grid, volume], nodes=1)
        self.assertIn("4 operations", fused.describe()[0])

    def test_index_names_and_launch_dimensions_differ(self):
        count, padded = wp.zeros(6, dtype=int), wp.zeros((9, 2), dtype=int)
        sequence = _Sequence()
        sequence.launch(_leading, (6, 5), [count])
        sequence.launch(_leading, (4, 3, 2), [count])
        sequence.launch(_padded, 9, [padded])
        self._check(sequence, [count, padded], nodes=1)

    def test_launch_without_threads_issues_nothing(self):
        line, empty = wp.zeros(12, dtype=int), wp.zeros(0, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 12, [line, 2])
        sequence.launch(_ramp, 12, [line, 5])
        fused = self._check(sequence, [line], nodes=1)
        operations = list(self._recorded(sequence))
        operations.insert(1, Launch(_ramp, (0,), (empty, 9), (), DEFAULT_BLOCK))
        operations.append(Launch(_grid, (4, 0), (wp.zeros((4, 0), dtype=int),), (), DEFAULT_BLOCK))
        with_empty = FusedLaunches(operations, 1, DEVICE)
        self.assertEqual(with_empty.node_count, 1)
        self.assertEqual(with_empty.describe(), fused.describe())
        self.assertEqual(FusedLaunches([operations[1]], 1, DEVICE).node_count, 0)
        if CUDA:
            body = _Sequence()
            body.launch(_ramp, 0, [empty, 9])
            self.assertEqual(record(body, DEVICE), [])

    def test_kernel_without_thread_index(self):
        counter, line = wp.zeros(1, dtype=int), wp.zeros(5, dtype=int)
        sequence = _Sequence()
        sequence.launch(_count, 1, [counter])
        sequence.launch(_ramp, 5, [line, 2])
        sequence.launch(_count, 11, [counter])
        sequence.launch(_count, (3, 4), [counter])
        self._check(sequence, [counter, line], nodes=1)
        self.assertEqual(counter.numpy()[0], 0)

    def test_early_returns_in_nested_loops(self):
        table = np.full((40, 6), 999, dtype=np.int32)
        table[::4, 3] = 32  # matches j 3, k 2
        table[1::4, 5] = 50  # matches j 5, k 0
        table[2::4, 0] = -4  # returns before the loops
        table, out = wp.array(table, dtype=int), wp.zeros(40, dtype=int)
        sequence = _Sequence()
        sequence.launch(_first_match, 40, [table, out])
        sequence.launch(_ramp, 40, [wp.zeros(40, dtype=int), 1])
        self._check(sequence, [out], nodes=1)
        fused = FusedLaunches(self._recorded(sequence), 1, DEVICE)
        fused.launch()
        self.assertEqual(sorted(set(out.numpy().tolist())), [-2, -1, 302, 500])

    def test_closure_constants_and_static_expressions(self):
        double, triple = _scaled(2, 5), _scaled(3, 7)
        self.assertIsNot(double, triple)
        self.assertEqual(double.key, triple.key)
        results, names = [], []
        for order in ((double, triple), (triple, double), (double, double)):
            values = _ints(21, seed=4)
            sequence = _Sequence()
            for kernel in order:
                sequence.launch(kernel, 21, [values])
            fused = self._check(sequence, [values], nodes=1)
            fused.launch()
            results.append(values.numpy().copy())
            names.append(_kernel_name(fused))
        self.assertEqual(len(set(names)), 3)
        self.assertFalse(np.array_equal(results[0], results[1]))
        self.assertFalse(np.array_equal(results[0], results[2]))

    def test_kernels_of_two_factories_with_one_name_and_source(self):
        ours, theirs = _scaled(2, 5), _scaled_twice(2, 5)
        self.assertEqual((ours.key, ours.adj.source), (theirs.key, theirs.adj.source))
        self.assertIs(_scaled_twice(1, 5), ours)
        values = _ints(15, seed=6)
        sequence = _Sequence()
        sequence.launch(ours, 15, [values])
        sequence.launch(theirs, 15, [values])
        sequence.launch(ours, 15, [values])
        fused = self._check(sequence, [values], nodes=1)
        self.assertEqual(len(set(re.findall(r"scaled \(15,\)", fused.describe()[0]))), 1)
        expected = values.numpy().copy()
        fused.launch()
        for scale in (2, 4, 2):
            expected = expected * scale + 15 + np.arange(15)
        np.testing.assert_array_equal(values.numpy(), expected)

    def test_atomics(self):
        keys = wp.array(np.random.default_rng(2).integers(0, 9, (13, 17)), dtype=int)
        bins, values = wp.zeros(9, dtype=int), _ints(300, seed=3)
        count, slots, total = wp.zeros(1, dtype=int), wp.zeros(300, dtype=int), wp.zeros(9, dtype=int)
        sequence = _Sequence()
        sequence.launch(_histogram, (13, 17), [keys], [bins])
        sequence.launch(_accumulate, 9, [bins, bins, total])
        sequence.launch(_histogram, (13, 17), [keys, bins])
        self._check(sequence, [bins, total], nodes=1)

        appending = _Sequence()
        appending.launch(_append, 300, [values, count, slots])
        appending.launch(_ramp, 9, [bins, 1])
        appending()
        expected = (int(count.numpy()[0]), sorted(slots.numpy()[: count.numpy()[0]].tolist()))
        self.assertGreater(expected[0], 50)
        operations = self._recorded(appending)
        count.zero_()
        slots.zero_()
        FusedLaunches(operations, 1, DEVICE).launch()
        self.assertEqual((int(count.numpy()[0]), sorted(slots.numpy()[: count.numpy()[0]].tolist())), expected)

    def test_stage_reads_what_other_threads_of_the_stage_before_wrote(self):
        values, out = _ints(256, seed=1), wp.zeros(256, dtype=int)
        sequence = _Sequence()
        sequence.launch(_slow_write, 256, [values])
        sequence.launch(_mirror, 256, [values, out])
        sequence.launch(_slow_write, 256, [out])
        sequence.launch(_mirror, 256, [out], [values])
        fused = self._check(sequence, [values, out], nodes=1)
        # two arrays and nothing else, however often a stage takes them
        self.assertEqual(_arguments(fused), 2)

    def test_scalars_keep_their_recorded_values(self):
        line, floats = wp.zeros(10, dtype=int), wp.zeros(10, dtype=float)
        sequence = _Sequence()
        sequence.launch(_ramp, 10, [line, 2])
        sequence.launch(_generic_add, 10, [floats, 2.5])
        sequence.launch(_add_floats, 10, [floats, line])
        sequence.launch(_generic_add, 10, [floats, 0.75])
        sequence.launch(_generic_add, 10, [line, 3])
        sequence.launch(_add_floats, 10, [floats, line])
        fused = self._check(sequence, [line, floats], nodes=1)
        # two arrays and one argument per scalar of the six launches
        self.assertEqual(_arguments(fused), 2 + 4)

    def test_equal_sequences_share_a_generated_kernel(self):
        names = []
        for _ in range(2):
            line, total = wp.zeros(8, dtype=int), wp.zeros(8, dtype=int)
            sequence = _Sequence()
            sequence.launch(_ramp, 8, [line, 2])
            sequence.launch(_accumulate, 8, [line, line, total])
            names.append(_kernel_name(self._check(sequence, [line, total], nodes=1)))
        self.assertEqual(names[0], names[1])
        longer = _Sequence()
        longer.launch(_ramp, 9, [wp.zeros(9, dtype=int), 2])
        longer.launch(_accumulate, 9, [line, line, total])
        self.assertNotEqual(_kernel_name(FusedLaunches(longer.operations, 1, DEVICE)), names[0])

    def test_generated_kernel_name_repeats_in_another_process(self):
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), NEWTON_TEST_DEVICE="cpu")
        command = "import test_fused_launches as t; print(t.probe_kernel_name())"
        result = subprocess.run([sys.executable, "-c", command], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with wp.ScopedDevice("cpu"):
            self.assertEqual(result.stdout.split()[-1], probe_kernel_name())

    def test_arrays_stay_alive(self):
        line, total, cleared = wp.zeros(8, dtype=int), wp.zeros(8, dtype=int), wp.ones(8, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 8, [line, 2])
        sequence.zero(cleared)
        sequence.launch(_accumulate, 8, [line, line, total])
        if CUDA:
            fused = FusedLaunches(record(sequence, DEVICE), 1, DEVICE)
        else:
            fused = FusedLaunches(sequence.operations, 1, DEVICE)
        alive = [weakref.ref(array) for array in (line, cleared)]
        del sequence, line, cleared
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in alive))
        fused.launch()
        np.testing.assert_array_equal(total.numpy(), 3 * (2 * np.arange(8) + 1))
        np.testing.assert_array_equal(alive[1]().numpy(), 0)
        del fused
        gc.collect()
        self.assertTrue(all(reference() is None for reference in alive))


def probe_kernel_name():
    """Name of the generated kernel of a fixed sequence on the current device."""
    line, grid = wp.zeros(8, dtype=int), wp.zeros((2, 3), dtype=int)
    sequence = _Sequence()
    sequence.launch(_ramp, 8, [line, 2])
    sequence.launch(_scaled(4, 9), 8, [line])
    sequence.zero(grid)
    sequence.launch(_grid, (2, 3), [grid])
    return _kernel_name(FusedLaunches(sequence.operations, 1, wp.get_device()))


class MemoryTests(_SequenceTestCase):
    def test_fills_and_copies_between_launches(self):
        count = 40
        first, second, total = wp.zeros(count, dtype=int), wp.zeros(count, dtype=int), wp.zeros(count, dtype=int)
        small, floats, longs = wp.zeros(33, dtype=wp.uint8), wp.zeros(count, dtype=float), wp.zeros(8, dtype=wp.int64)
        shorts = wp.zeros(9, dtype=wp.int16)
        sequence = _Sequence()
        sequence.launch(_ramp, count, [first, 2])
        sequence.copy(second, first)
        sequence.zero(first)
        sequence.launch(_accumulate, count, [first, second, total])
        sequence.launch(_ramp, count, [first, 3])
        sequence.fill(first, 5)  # elements of four different bytes
        sequence.launch(_accumulate, count, [first, second, total])
        sequence.copy(second, first, dest_offset=3, src_offset=1, count=6)
        sequence.launch(_accumulate, count, [first, second, total])
        sequence.fill(small, 7)
        sequence.launch(_add_bytes, 33, [small, total])
        sequence.zero(small)
        sequence.fill(floats, 2.75)
        sequence.launch(_add_floats, count, [floats, total])
        sequence.zero(floats)
        sequence.launch(_add_floats, count, [floats, total])
        sequence.fill(longs, 123456789012345)
        sequence.launch(_add_longs, 8, [longs, total])
        sequence.fill(shorts, -2)  # elements of two equal bytes
        sequence.launch(_add_shorts, 9, [shorts, total])
        sequence.fill(shorts, 513)
        sequence.launch(_add_shorts, 9, [shorts, total])
        sequence.fill(first, 0)  # elements of four equal bytes
        sequence.launch(_accumulate, count, [first, second, total])
        fused = self._check(sequence, [first, second, total, small, floats, longs, shorts], nodes=1)
        described = fused.describe()[0]
        for stage in ("memcpy 160 bytes", "memset 160 bytes", "memcpy 24 bytes", "memset 33 bytes", "memset 64 bytes"):
            self.assertIn(stage, described)
        # the seven arrays and two scalars of the launches, the bytes of five arrays and of the two ranges
        # of the partial copy (the bytes of the byte array are that array), and the value of every fill
        self.assertEqual(_arguments(fused), 7 + 2 + 5 + 2 + 10)

    def test_memory_operations_alone(self):
        first, second = _ints(12, seed=8), wp.zeros(12, dtype=int)
        sequence = _Sequence()
        sequence.copy(second, first)
        sequence.fill(first, 3)
        fused = self._check(sequence, [first, second], nodes=1)
        self.assertIn("2 operations", fused.describe()[0])
        single = _Sequence()
        single.fill(first, 4)
        self.assertIn("is alone", self._check(single, [first], nodes=1).describe()[0])
        copy = _Sequence()
        copy.copy(second, first, dest_offset=2, count=5)
        self.assertIn("is alone", self._check(copy, [first, second], nodes=1).describe()[0])

    def test_large_operations_keep_their_nodes(self):
        items, size = fused_launches.STAGE_ITEMS_MAX, fused_launches.MEMORY_BYTES_MAX
        large, total = wp.zeros(items + 1, dtype=int), wp.zeros(items + 1, dtype=int)
        wide, other = wp.ones(size + 1, dtype=wp.uint8), wp.zeros(size + 1, dtype=wp.uint8)
        sequence = _Sequence()
        sequence.launch(_ramp, items, [large, 2])
        sequence.launch(_ramp, items, [total, 3])
        sequence.launch(_ramp, items + 1, [large, 4])
        sequence.launch(_accumulate, items, [large, large, total])
        sequence.zero(wide)
        sequence.copy(other, wide, count=size)
        sequence.launch(_add_bytes, items, [wide, total])
        sequence.fill(wide, 3)
        sequence.copy(other, wide)
        sequence.launch(_add_bytes, items, [other, total])
        # fused: the first two launches, and the copy of the largest size with the launch after it
        fused = self._check(sequence, [large, total, wide, other], nodes=8)
        described = fused.describe()
        self.assertIn(f"has {items + 1} items per block", described[1])
        for node in (3, 5, 6):
            self.assertIn(f"has {size + 1} bytes", described[node])
        for node in (2, 7):
            self.assertIn("is alone between operations that are not fused", described[node])
        self.assertIn(f"memcpy {size} bytes", described[4])

    def test_arguments_beyond_the_launch_limit_start_a_new_kernel(self):
        arrays = [wp.zeros(6, dtype=int) for _ in range(12)]
        sequence = _Sequence()
        for index, array in enumerate(arrays):
            sequence.launch(_ramp, 6, [array, index])
        self._check(sequence, arrays, nodes=1)
        # an array and a scalar take 64 bytes: room for five launches per kernel
        with mock.patch.object(fused_launches, "_ARGUMENT_BYTES_MAX", 5 * 64):
            self._check(sequence, arrays, nodes=3)

    @unittest.skipUnless(CUDA, CUDA_ONLY)
    def test_temporary_arrays_of_a_pool(self):
        scratch, total = StepScratch(DEVICE), wp.zeros(20, dtype=int)

        def body():
            temporary = wp.zeros(20, dtype=int)
            wp.launch(_ramp, 20, [temporary, 3])
            wp.launch(_accumulate, 20, [temporary, temporary, total])
            cleared = wp.zeros((2, 10), dtype=int)
            cleared[1].fill_(4)
            wp.launch(_accumulate, 20, [temporary, cleared.flatten(), total])

        with scratch.scope():
            body()
        expected = total.numpy().copy()
        total.zero_()
        with scratch.scope():
            operations = record(body, DEVICE)
        self.assertEqual(
            [type(operation).__name__ for operation in operations],
            ["Memset", "Launch", "Launch", "Memset", "Memset", "Launch"],
        )
        self.assertEqual([operation.count for operation in operations if isinstance(operation, Memset)], [80, 80, 10])
        fused = FusedLaunches(operations, 1, DEVICE)
        self.assertEqual(fused.node_count, 1)
        with allocation_tracker() as tracker:
            fused.launch()
            self.assertEqual(allocations(tracker), 0)
        np.testing.assert_array_equal(total.numpy(), expected)


class BoundaryTests(_SequenceTestCase):
    def test_tile_kernel_keeps_its_launch(self):
        values = wp.array(np.arange(5 * TILE, dtype=np.float32).reshape(5, TILE))
        sums, total, line = wp.zeros(5, dtype=float), wp.zeros(5, dtype=int), wp.zeros(5, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 5, [line, 2])
        sequence.launch(_accumulate, 5, [line, line, total])
        sequence.launch_tiled(_tile_sum, [5], [values, sums], block_dim=64)
        sequence.launch(_add_floats, 5, [sums, total])
        sequence.launch(_ramp, 5, [line, 3])
        fused = self._check(sequence, [sums, total, line], nodes=3)
        self.assertIn("works across the threads of a block (tile_load)", fused.describe()[1])

    def test_block_striding_kernel_keeps_its_launch(self):
        values = wp.array(np.arange(3 * 50).reshape(3, 50), dtype=int)
        scratch, line = wp.zeros((3, 50), dtype=int), wp.zeros(5, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 5, [line, 2])
        sequence.launch(_ramp, 5, [line, 3])
        sequence.launch_tiled(_block_reverse, [3], [values, scratch], block_dim=32)
        sequence.launch_tiled(_block_reverse, [3], [values, scratch], block_dim=32)
        sequence.launch_tiled(_block_reverse, [3], [values, scratch], block_dim=32)
        fused = self._check(sequence, [values, line], nodes=4)
        self.assertIn("works across the threads of a block", fused.describe()[1])

    def test_native_code_behind_a_function_keeps_its_launch(self):
        line = wp.zeros(256, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 256, [line, 2])
        sequence.launch(_ramp, 256, [line, 3])
        sequence.launch(_calls_native, 256, [line])
        sequence.launch(_ramp, 5, [line, 3])
        fused = self._check(sequence, [line], nodes=3)
        self.assertIn("works across the threads of a block (native function", fused.describe()[1])

    def test_thread_index_that_cannot_be_passed_keeps_its_launch(self):
        line = wp.zeros(9, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 9, [line, 2])
        sequence.launch(_ramp, 9, [line, 3])
        sequence.launch(_expression_index, 9, [line])
        sequence.launch(_nested_index, 9, [line])
        sequence.launch(_ramp, 9, [line, 4])
        sequence.launch(_expression_index, 9, [line])
        fused = self._check(sequence, [line], nodes=5)
        for node in (1, 2, 4):
            self.assertIn("reads the thread index other than by one wp.tid()", fused.describe()[node])

    def test_kernel_and_module_options(self):
        line = wp.zeros(9, dtype=int)
        sequence = _Sequence()
        sequence.launch(_ramp, 9, [line, 2])
        sequence.launch(_ramp, 9, [line, 3], block_dim=64, max_blocks=3)  # how a launch is spread changes no stage
        sequence.launch(_fast_math_add, 9, [line])
        sequence.launch(_fast_math_add, 9, [line])
        sequence.launch(_ramp, 9, [line, 4])
        sequence.launch(_register_limited, 9, [line])
        sequence.launch(_ramp, 9, [line, 5])
        fused = self._check(sequence, [line], nodes=5)
        described = fused.describe()
        self.assertIn("2 operations", described[0])
        self.assertIn("_fast_math_add (9,); _fast_math_add (9,)", described[1])
        self.assertIn("has launch options", described[3])

        deterministic = [Launch(_deterministic_add, (9,), (line,)), Launch(_deterministic_add, (9,), (line,))]
        described = FusedLaunches(deterministic, 1, DEVICE).describe()
        self.assertEqual(len(described), 2)
        self.assertIn("is in a module with deterministic atomics", described[0])


class WorldTests(_SequenceTestCase):
    def _sequence(self, worlds):
        count = 70
        values, out = wp.zeros((worlds, count), dtype=int), wp.zeros((worlds, count), dtype=int)
        total = wp.zeros(worlds, dtype=int)
        sequence = _Sequence()
        sequence.launch(_world_ramp, (worlds, count), [values, 3])
        sequence.launch(_world_mirror, (worlds, count), [values, out])
        sequence.launch(_world_total, worlds, [out, total])
        sequence.launch(_pairs, worlds * count, [values, out])
        sequence.launch(_world_mirror, (worlds, count), [out, values])
        sequence.launch(_world_total, worlds, [values, total])
        sequence.launch(_next_world, (worlds, count), [values, out])
        sequence.launch(_world_ramp, (worlds, count), [values, 5])
        sequence.launch(_world_mirror, (worlds, count), [values, out])
        return sequence, [values, out, total]

    def test_one_world_takes_every_launch(self):
        sequence, arrays = self._sequence(1)
        self._check(sequence, arrays, nodes=1)

    def test_several_worlds_fuse_the_launches_led_by_the_world_index(self):
        sequence, arrays = self._sequence(3)
        fused = self._check(sequence, arrays, nworld=3, nodes=5)
        described = fused.describe()
        self.assertIn("3 operations", described[0])
        self.assertIn("_pairs (210,): is not led by the world index", described[1])
        self.assertIn("2 operations", described[2])
        self.assertIn("_next_world (3, 70): is not led by the world index", described[3])
        self.assertIn("2 operations", described[4])

    def test_launch_over_some_worlds_is_not_led_by_the_world_index(self):
        values, out, total = wp.zeros((4, 9), dtype=int), wp.zeros((4, 9), dtype=int), wp.zeros(4, dtype=int)
        sequence = _Sequence()
        sequence.launch(_world_ramp, (4, 9), [values, 3])
        sequence.launch(_world_total, 4, [values, total])
        sequence.launch(_world_ramp, (2, 9), [values, 7])
        sequence.launch(_world_mirror, (4, 9), [values, out])
        sequence.launch(_world_total, 4, [out, total])
        fused = self._check(sequence, [values, out, total], nworld=4, nodes=3)
        self.assertIn("_world_ramp (2, 9): is not led by the world index", fused.describe()[1])

    def test_memory_operations_of_several_worlds_keep_their_nodes(self):
        values, out, total = wp.zeros((3, 9), dtype=int), wp.zeros((3, 9), dtype=int), wp.zeros(3, dtype=int)
        sequence = _Sequence()
        sequence.launch(_world_ramp, (3, 9), [values, 3])
        sequence.launch(_world_mirror, (3, 9), [values, out])
        sequence.copy(values, out)
        sequence.zero(out)
        sequence.launch(_world_mirror, (3, 9), [values, out])
        sequence.launch(_world_total, 3, [out, total])
        fused = self._check(sequence, [values, out, total], nworld=3, nodes=4)
        self.assertIn("is not the work of one world", fused.describe()[1])
        self.assertIn("is not the work of one world", fused.describe()[2])
        one = _Sequence()
        one.operations = [operation for operation in sequence.operations]
        self.assertEqual(FusedLaunches(one.operations, 1, DEVICE).node_count, 1)


@unittest.skipUnless(CUDA, CUDA_ONLY)
class CaptureTests(_SequenceTestCase):
    def _sequence(self):
        values, out, total = _ints(64, seed=5), wp.zeros(64, dtype=int), wp.zeros(64, dtype=int)
        sequence = _Sequence()
        sequence.launch(_slow_write, 64, [values])
        sequence.launch(_mirror, 64, [values, out])
        sequence.copy(values, out)
        sequence.launch(_accumulate, 64, [values, out, total])
        sequence.launch_tiled(_block_reverse, [1], [total.reshape((1, 64)), out.reshape((1, 64))], block_dim=32)
        sequence.fill(out, 3)
        sequence.launch(_accumulate, 64, [values, out, total])
        return sequence, [values, out, total]

    def _expected(self, sequence, arrays, repeats):
        start = [array.numpy().copy() for array in arrays]
        for _ in range(repeats):
            sequence()
        expected = [array.numpy().copy() for array in arrays]
        for array, value in zip(arrays, start, strict=True):
            array.assign(value)
        return expected

    def test_replays(self):
        sequence, arrays = self._sequence()
        expected = self._expected(sequence, arrays, 3)
        fused = FusedLaunches(record(sequence, DEVICE), 1, DEVICE)
        self.assertEqual(fused.node_count, 3)
        with wp.ScopedCapture() as capture:
            fused.launch()
        for _ in range(3):
            wp.capture_launch(capture.graph)
        self._assert_arrays(arrays, expected)

    def test_body_of_a_graph_loop(self):
        sequence, arrays = self._sequence()
        expected = self._expected(sequence, arrays, 4)
        fused = FusedLaunches(record(sequence, DEVICE), 1, DEVICE)
        remaining = wp.zeros(1, dtype=int)

        def body():
            fused.launch()
            wp.launch(count_down, 1, [remaining])

        start = [array.numpy().copy() for array in arrays]
        remaining.fill_(4)
        wp.capture_while(remaining, body)  # outside capture Warp runs the loop on the host
        self._assert_arrays(arrays, expected)
        for array, value in zip(arrays, start, strict=True):
            array.assign(value)
        with wp.ScopedCapture() as capture:
            wp.capture_while(remaining, body)
        self._assert_arrays(arrays, start)
        remaining.fill_(4)
        wp.capture_launch(capture.graph)
        self._assert_arrays(arrays, expected)

    def test_branch_of_a_graph_conditional(self):
        sequence, arrays = self._sequence()
        expected = self._expected(sequence, arrays, 1)
        fused = FusedLaunches(record(sequence, DEVICE), 1, DEVICE)
        condition = wp.zeros(1, dtype=int)
        start = [array.numpy().copy() for array in arrays]
        with wp.ScopedCapture() as capture:
            wp.capture_if(condition, on_true=fused.launch)
        wp.capture_launch(capture.graph)
        self._assert_arrays(arrays, start)
        condition.fill_(1)
        wp.capture_launch(capture.graph)
        self._assert_arrays(arrays, expected)

    def test_launch_allocates_nothing(self):
        sequence, _arrays = self._sequence()
        fused = FusedLaunches(record(sequence, DEVICE), 1, DEVICE)
        fused.launch()
        with allocation_tracker() as tracker:
            fused.launch()
            self.assertEqual(allocations(tracker), 0)


class ErrorTests(DeviceTestCase):
    def test_record_needs_a_cuda_device(self):
        with self.assertRaisesRegex(RuntimeError, "cannot run on device 'cpu'"):
            record(lambda: None, "cpu")

    def test_unknown_records_and_values(self):
        line = wp.zeros(4, dtype=int)
        with self.assertRaisesRegex(ValueError, "takes Launch, Memset and Memcpy, got tuple"):
            FusedLaunches([(_ramp, 4)], 1, DEVICE)
        with self.assertRaisesRegex(ValueError, "_ramp with 1 values for 2 arguments"):
            FusedLaunches([Launch(_ramp, (4,), (line,))], 1, DEVICE)
        with self.assertRaisesRegex(ValueError, "elements of 3 bytes"):
            FusedLaunches([Memset(line.ptr, 0, 3, 4, (line,))], 1, DEVICE)
        with self.assertRaisesRegex(ValueError, "at least one world"):
            FusedLaunches([], 0, DEVICE)
        self.assertEqual(FusedLaunches([], 1, DEVICE).node_count, 0)

    @unittest.skipUnless(CUDA, CUDA_ONLY)
    def test_record_rejects_what_it_cannot_list(self):
        line, vectors = wp.zeros(8, dtype=int), wp.zeros(8, dtype=wp.vec3)
        remaining = wp.ones(1, dtype=int)

        def allocates():
            wp.launch(_ramp, 8, [wp.zeros(8, dtype=int), 2])

        def loops():
            wp.capture_while(remaining, lambda: wp.launch(count_down, 1, [remaining]))

        def untaped():
            wp.launch(_ramp, 8, [line, 2])
            wp.launch(_ramp, 8, [line, 3], record_tape=False)

        def wide_fill():
            vectors.fill_(wp.vec3(1.0, 2.0, 3.0))

        def from_host():
            line.assign(np.arange(8, dtype=np.int32))

        def other_device():
            wp.launch(_ramp, 8, [wp.zeros(8, dtype=int, device="cpu"), 2], device="cpu")

        for body, message in (
            (allocates, "memory allocation node"),
            (loops, "conditional node"),
            (untaped, "not the next launch of the tape"),
            (wide_fill, "not the next launch of the tape"),
            (from_host, "host memory"),
            (other_device, "on device 'cpu' instead of"),
        ):
            with self.subTest(body=body.__name__), self.assertRaisesRegex(RuntimeError, message):
                record(body, DEVICE)
        self.assertFalse(wp.get_device().is_capturing)
        with wp.ScopedCapture():
            with self.assertRaisesRegex(RuntimeError, "cannot run during graph capture"):
                record(lambda: None, DEVICE)

    @unittest.skipUnless(CUDA, CUDA_ONLY)
    def test_graph_nodes_of_a_branching_graph(self):
        remaining = wp.ones(1, dtype=int)
        with wp.ScopedCapture() as capture:
            wp.capture_if(remaining, on_true=lambda: wp.launch(count_down, 1, [remaining]))
        with self.assertRaisesRegex(RuntimeError, "conditional node|not one chain"):
            graph_nodes(capture.graph)


# The stages of a step that the rigid step issues outside its constraint solve, by name.
def _kinematics(m, d):
    smooth.kinematics(m, d)
    smooth.com_pos(m, d)
    smooth.crb(m, d)


def _velocity(m, d):
    smooth.com_vel(m, d)
    passive.passive(m, d)
    smooth.rne(m, d)


def _contacts(m, d):
    collision_driver.collision(m, d)
    constraint.make_constraint(m, d)


# In the order of a step: each stage reads what the stages before it leave in Data.
MUJOCO_STAGES = {
    "kinematics": _kinematics,
    "contacts": _contacts,
    "velocity": _velocity,
    "actuation": forward.fwd_actuation,
    "implicit": forward.implicit,
}
# name of the scene, worlds
MUJOCO_CASES = (("arm", 1), ("gripper", 1), ("boxes", 1), ("humanoid", 1), ("cables", 1), ("gripper", 2))


def _data(d):
    return {name: array.numpy().copy() for name, array in data_arrays(d) if array.size}


@unittest.skipUnless(CUDA, CUDA_ONLY)
class MujocoWarpStageTests(DeviceTestCase):
    def _assert_stage(self, m, before, function, worlds):
        """Fused launches of one stage against two stock calls from equal Data; returns Data after the stage."""
        stock = [clone_data(before), clone_data(before)]
        for d in stock:
            function(m, d)
        expected, repeated = _data(stock[0]), _data(stock[1])
        start = _data(before)
        self.assertTrue(any(not np.array_equal(value, start[key], equal_nan=True) for key, value in expected.items()))

        scratch, warm, d = StepScratch(DEVICE), clone_data(before), clone_data(before)
        with scratch.scope():
            function(m, warm)  # sizes the pool
        with scratch.scope():
            operations = record(lambda: function(m, d), DEVICE)
        self.assertEqual(_data(d).keys(), start.keys())
        for key, value in _data(d).items():
            np.testing.assert_array_equal(value, start[key], err_msg=f"{key} after recording")
        fused = FusedLaunches(operations, worlds, DEVICE)
        fused.launch()
        result, exact = _data(d), 0
        for key, value in expected.items():
            if np.array_equal(value, repeated[key], equal_nan=True):
                np.testing.assert_array_equal(result[key], value, err_msg=key)
                exact += 1
            else:
                bound = 1e-6 * float(np.abs(value).max())
                np.testing.assert_allclose(result[key], value, rtol=0.0, atol=bound, err_msg=key)
        return stock[0], len(operations), fused.node_count, exact, len(expected)

    def test_stages_match_mujoco_warp(self):
        rng = np.random.default_rng(3)
        for name, worlds in MUJOCO_CASES:
            m, d = step_model(name, worlds)
            # a step leaves Data of the device and another position; other velocities and controls in
            # every world, and forces where a scene without actuators only clears them, give every
            # stage new results to compute
            mjw.step(m, d)
            for field in (d.qvel, d.ctrl):
                value = field.numpy()
                field.assign(value + rng.uniform(-0.2, 0.2, value.shape).astype(np.float32))
            d.qfrc_actuator.fill_(0.5)
            recorded = issued = 0
            for stage, function in MUJOCO_STAGES.items():
                if stage == "implicit":
                    mjw.forward(m, d)  # the acceleration that the integrator advances with
                with self.subTest(model=name, worlds=worlds, stage=stage):
                    d, operations, nodes, _exact, _arrays = self._assert_stage(m, d, function, worlds)
                    recorded += operations
                    issued += nodes
            with self.subTest(model=name, worlds=worlds):
                self.assertLess(2 * issued, recorded)


if __name__ == "__main__":
    unittest.main()
