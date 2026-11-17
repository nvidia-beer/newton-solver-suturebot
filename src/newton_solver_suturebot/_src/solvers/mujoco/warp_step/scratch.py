# SPDX-License-Identifier: Apache-2.0
"""Persistent device buffers for the temporary arrays of a MuJoCo Warp step.

``mujoco_warp.step`` creates its temporary arrays (collision context, constraint
and transmission counters, solver context, integrator scratch) on every call.
In a CUDA graph capture each of them becomes a memory allocation node and a
memory free node of the graph, and Warp refuses to capture the body of a
conditional graph node that allocates memory. ``StepScratch`` is a Warp
allocator (``wp.Allocator``) that serves those requests from buffers it keeps,
so a captured step holds no allocation nodes and an eager step asks the device
for no memory. MuJoCo Warp is called as it is: ``wp.ScopedAllocator`` installs
the allocator on the device for the duration of a scope.
"""

import itertools
import threading
from contextlib import contextmanager

import warp as wp

_serial = itertools.count()


class StepScratch:
    """Warp allocator that gives the k-th array created in a scope the k-th buffer of a pool.

    Use, with the same calls in every scope::

        scratch = StepScratch(device)
        with scratch.scope():
            mujoco_warp.step(m, d)  # outside capture: the pool grows to what the step asks for
        with wp.ScopedCapture() as capture:
            with scratch.scope():
                mujoco_warp.step(m, d)  # the same requests take the same buffers

    Inside ``scope()`` every array created on the device (``wp.empty``, ``wp.zeros``,
    ``wp.full``, ``wp.clone``, their ``_like`` forms and ``wp.array`` from host data,
    whoever calls them) takes the next buffer of the pool, in creation order. The
    buffers are byte arrays from the built-in allocator of the device; they stay with
    the pool when the array that used them dies.

    What the caller must know:

    - Two arrays of one scope never share a buffer; arrays of different scopes do. A
      buffer holds what the previous scope left in it, which is the contract of
      ``wp.empty``; ``wp.zeros`` and ``wp.full`` initialize it as always. Work of two
      scopes of one pool must not run concurrently on the device, for example in two
      branches of one graph.
    - One pool serves one sequence of requests, which must not depend on the data.
      Outside capture a request past the end of the pool adds a buffer, and a request
      of another size than the buffer at its position replaces that buffer. During
      capture both raise ``RuntimeError``: the built-in allocator would add allocation
      nodes to the graph, and graphs captured earlier address the buffers they were
      captured with. Outside capture ``wp.capture_while`` runs its body once per
      iteration and a capture records it once, so a loop body that creates arrays asks
      for a sequence that the capture does not repeat. MuJoCo Warp's Newton iteration
      creates none, unless ``opt.deterministic`` asks for deterministic atomics: Warp
      then creates the buffers of every such launch, with sizes that follow the data.
    - A captured graph addresses the buffers by pointer. Keep the pool alive, and do
      not close it, while such a graph is launched, and put ``signature`` into the key
      of a graph cache: it changes whenever a buffer is added or replaced. A replaced
      buffer is not freed before ``close()``, so a graph captured before the
      replacement still addresses valid memory; a later replacement of the same size
      takes it into use again. A pool that is asked for new sizes in every scope grows
      until ``close()``.
    - An array that outlives its scope keeps its buffer for itself: the next request
      at that position replaces the buffer (outside capture) or raises (during
      capture). This covers arrays that the scoped code stores, arrays in reference
      cycles until the garbage collector frees them, and arrays in the frames of a
      traceback after an exception left the scope. The scope itself ends cleanly on an
      exception: the previous allocator is restored and the next scope starts at the
      first buffer.
    - The allocator is installed for the whole device. An array that another thread
      creates on the device while a scope is open raises ``RuntimeError`` in that
      thread instead of taking a buffer of the pool.
    - Memory that Warp allocates natively passes through no allocator and is not
      served here. ``wp.utils.array_scan`` allocates a temporary buffer on every call,
      and MuJoCo Warp's sweep-and-prune broadphase calls it once per step.
    - On a CPU device ``scope()`` installs nothing, because Warp supports custom
      allocators on CUDA devices only; arrays are allocated as without it.
    """

    # deallocate() makes no CUDA call, so a dying array need not make the device context current for it.
    deallocate_requires_context_guard = False

    def __init__(self, device):
        self.device = wp.get_device(device)
        self._serial = next(_serial)
        self._version = 0
        self._buffers = []  # the buffer of each position
        self._replaced = {}  # size in bytes -> buffers that left their position
        self._held = {}  # pointer -> buffer, for every living array that took one
        self._position = None  # position of the next request; None outside a scope
        self._thread = None

    @property
    def buffer_count(self):
        """Number of buffers that a scope hands out."""
        return len(self._buffers)

    @property
    def signature(self):
        """Hashable identity of the buffers that a scope hands out, for graph cache keys.

        Changes whenever a buffer is added or replaced and when the pool is closed. Two pools
        never have equal signatures.
        """
        return (self._serial, self._version)

    @contextmanager
    def scope(self):
        """Serve the arrays that the block creates on the device, starting at the first buffer."""
        if self._position is not None:
            raise RuntimeError("StepScratch scopes do not nest")
        self._position = 0
        self._thread = threading.get_ident()
        try:
            if self.device.is_cuda:
                with wp.ScopedAllocator(self.device, self):
                    yield
            else:
                yield
        finally:
            self._position = None
            self._thread = None

    def allocate(self, size_in_bytes):
        """Return the pointer of the buffer at the next position of the open scope."""
        position = self._position
        if position is None:
            raise RuntimeError("StepScratch serves requests only inside scope()")
        if threading.get_ident() != self._thread:
            raise RuntimeError(f"A StepScratch scope of another thread is open on device {self.device}")
        buffer = self._buffers[position] if position < len(self._buffers) else None
        if buffer is None or buffer.capacity != size_in_bytes or buffer.ptr in self._held:
            if self.device.is_capturing:
                if buffer is None:
                    present = "no buffer"
                elif buffer.capacity != size_in_bytes:
                    present = f"a buffer of {buffer.capacity} bytes"
                else:
                    present = f"a buffer of {buffer.capacity} bytes that an array of an earlier scope still holds"
                raise RuntimeError(
                    f"StepScratch request {position} for {size_in_bytes} bytes during graph capture, "
                    f"but the pool has {present} at that position; "
                    "run the scope with the same calls outside capture first"
                )
            replaced, buffer = buffer, self._unused_buffer(size_in_bytes)
            if replaced is None:
                self._buffers.append(buffer)
            else:
                self._replaced.setdefault(replaced.capacity, []).append(replaced)
                self._buffers[position] = buffer
            self._version += 1
        self._held[buffer.ptr] = buffer
        self._position = position + 1
        return buffer.ptr

    def deallocate(self, ptr, size_in_bytes):
        """Take the buffer back from an array that died; the memory stays allocated."""
        self._held.pop(ptr, None)

    def close(self):
        """Release the buffers. A buffer that a living array holds is released when that array dies."""
        if self._position is not None:
            raise RuntimeError("StepScratch cannot be closed inside scope()")
        self._buffers.clear()
        self._replaced.clear()
        self._version += 1

    def _unused_buffer(self, size_in_bytes):
        """A replaced buffer of this size that no array holds, else a new one from the built-in allocator."""
        replaced = self._replaced.get(size_in_bytes, ())
        for index, buffer in enumerate(replaced):
            if buffer.ptr not in self._held:
                return replaced.pop(index)
        with wp.ScopedAllocator(self.device, None):
            return wp.empty(size_in_bytes, dtype=wp.uint8, device=self.device)
