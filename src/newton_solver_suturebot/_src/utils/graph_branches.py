"""Concurrent branches inside one captured CUDA graph.

Work captured on one stream forms a chain: every node depends on the node before it,
so independent stages still run one after the other. The CUDA graph executor runs
nodes concurrently when the graph says they are independent, and a capturing stream
lets its owner choose what the next node depends on. ``Fork`` uses that: each branch
starts from the nodes captured before the fork instead of from the previous branch,
and the nodes after ``join()`` depend on the ends of all branches.

Only the dependency edges change. The same kernels run with the same inputs, so
branches must not write memory that another branch reads or writes.

Warp captures a conditional body by pausing the capture of the parent graph and
resuming it afterwards from every leaf of that graph, which includes the ends of the
branches captured earlier. A later branch therefore calls ``isolate()`` after each
conditional it captures at the level of the fork; conditionals nested inside a
conditional body need nothing. The first branch needs no calls.

The capture dependencies are read and replaced with the CUDA driver functions
``cuStreamGetCaptureInfo_v2`` and ``cuStreamUpdateCaptureDependencies``, which Warp
does not expose. ``Fork.begin`` returns ``None`` when they are unavailable, when the
device is not capturing, or on the CPU; callers then capture their stages in sequence.
"""

import ctypes
from contextlib import contextmanager

import warp as wp

_SET_CAPTURE_DEPENDENCIES = 1
_CAPTURE_STATUS_ACTIVE = 1
_driver = None


def _load_driver():
    """The CUDA driver library Warp already uses in this process, or False."""
    global _driver
    if _driver is None:
        try:
            library = ctypes.CDLL("libcuda.so.1")
            library.cuStreamGetCaptureInfo_v2.restype = ctypes.c_int
            library.cuStreamGetCaptureInfo_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            library.cuStreamUpdateCaptureDependencies.restype = ctypes.c_int
            library.cuStreamUpdateCaptureDependencies.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            _driver = library
        except (OSError, AttributeError):
            _driver = False
    return _driver


class Fork:
    """Capture independent stages of one graph so that they execute concurrently."""

    def __init__(self, device, stream, driver, start):
        self.device, self.stream, self._driver = device, stream, driver
        self._start = start
        self._ends = []
        self._branches = 0
        self._open = False
        self._joined = False

    @classmethod
    def begin(cls, device):
        """Start a fork at the current point of an active CUDA capture, or return ``None``."""
        device = wp.get_device(device)
        if not device.is_cuda or not device.is_capturing:
            return None
        driver = _load_driver()
        if not driver:
            return None
        stream = wp.get_stream(device)
        fork = cls(device, stream, driver, [])
        status, start = fork._capture()
        if status != _CAPTURE_STATUS_ACTIVE:
            return None
        fork._start = start
        return fork

    def _capture(self):
        status, identifier, graph = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
        nodes, count = ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_size_t()
        with self.device.context_guard:
            code = self._driver.cuStreamGetCaptureInfo_v2(
                ctypes.c_void_p(self.stream.cuda_stream),
                ctypes.byref(status),
                ctypes.byref(identifier),
                ctypes.byref(graph),
                ctypes.byref(nodes),
                ctypes.byref(count),
            )
        if code != 0:
            raise RuntimeError(f"CUDA capture dependencies are unavailable (driver error {code})")
        # The driver owns this list only until the next call on the stream.
        return status.value, [nodes[index] for index in range(count.value)]

    def _depend_on(self, nodes):
        array = (ctypes.c_void_p * len(nodes))(*nodes)
        with self.device.context_guard:
            code = self._driver.cuStreamUpdateCaptureDependencies(
                ctypes.c_void_p(self.stream.cuda_stream), array, len(nodes), _SET_CAPTURE_DEPENDENCIES
            )
        if code != 0:
            raise RuntimeError(f"CUDA capture dependencies could not be replaced (driver error {code})")

    @contextmanager
    def branch(self):
        """Capture one stage that depends only on what preceded the fork."""
        if self._open or self._joined:
            raise RuntimeError("Graph branches cannot be nested or added after the join")
        if wp.get_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("A graph branch must be captured on the stream that began the fork")
        self._open = True
        try:
            if self._branches:
                self._depend_on(self._start)
            self._branches += 1
            yield self
            self.isolate()
            self._ends.extend(node for node in self._capture()[1] if node not in self._ends)
        finally:
            self._open = False

    def isolate(self):
        """Drop the ends of earlier branches from the current dependencies.

        Call after every conditional captured directly in a branch other than the first.
        """
        if not self._open:
            raise RuntimeError("Graph branch dependencies can only be isolated inside a branch")
        if self._ends:
            nodes = [node for node in self._capture()[1] if node not in self._ends]
            # A branch that has captured nothing of its own yet still starts at the fork.
            self._depend_on(nodes if nodes else self._start)

    def join(self):
        """Make everything captured from here on depend on the ends of all branches."""
        if self._open or self._joined:
            raise RuntimeError("Join graph branches once, outside every branch")
        self._joined = True
        if self._ends:
            self._depend_on(self._ends)
