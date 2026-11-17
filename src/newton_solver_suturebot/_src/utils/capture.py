# SPDX-License-Identifier: Apache-2.0
"""Keep unrelated cyclic resource destruction outside CUDA graph capture, and count captures.

Warp arrays can own pinned memory. Collecting an old solver cycle may call
cudaFreeHost, which is not permitted during a CUDA capture. This scope delays
automatic Python cyclic GC, without changing reference counting or installing
hooks in Warp. Resource ownership must still prevent explicit frees in capture.

The scopes also keep a census of this process: the graphs captured through
:class:`ScopedCapture` on a CUDA device, the Krylov solves issued outside a capture
(stock Warp captures one single-use graph for each) and the kernel module builds that
are loaded. A scene ends its start-up with a mark (:func:`begin_startup`,
:func:`finish_startup`); :func:`counts` and :func:`modules` report what came after it.
The census wraps no Warp function and changes nothing in what is captured or launched.
"""

import gc
import os
import sys
import threading
import time
import weakref

import warp as wp
from warp._src import context as _warp_context

_lock = threading.RLock()
_depth = 0
_restore_gc = False
# Per thread: the number of open package captures, and the open GC pauses (innermost last).
_thread = threading.local()
# Source root of this package: capture sites inside it are named relative to it.
_PACKAGE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
# Names that a table of late capture sites or late kernel modules holds; the rest is counted under "other".
TABLE_NAMES = 16
# Keys of the start-up record of a mark (finish_startup).
STARTUP_RECORD = (
    "startup_seconds",
    "startup_captures",
    "startup_solver_captures",
    "startup_module_loads",
    "startup_unlaunched_graphs",
)


class Census:
    """Host counters of this process: integers that are read without walking anything."""

    def __init__(self):
        # Graphs captured through ScopedCapture on a CUDA device.
        self.captures = 0
        # Krylov solves issued outside a capture; stock Warp captures a single-use graph for each.
        self.solver_captures = 0
        # Captured graphs that their owners dropped (note_release).
        self.releases = 0
        # Call site -> captures and solver captures issued there.
        self.sites = {}
        # Live graphs captured through ScopedCapture on a CUDA device -> their call site.
        self.graphs = weakref.WeakKeyDictionary()


census = Census()


class Mark:
    """The census at one instant; ``startup`` is the record of the start-up that it ends."""

    __slots__ = ("captures", "modules", "releases", "sites", "solver_captures", "startup", "time")


def _site(frame):
    """Caller of a scope as ``path:line``; the path is relative to this package or to site-packages."""
    name = frame.f_code.co_filename
    if name.startswith(_PACKAGE):
        name = name[len(_PACKAGE) :]
    else:
        index = name.find("site-packages/")
        if index >= 0:
            name = name[index + 14 :]
    return f"{name}:{frame.f_lineno}"


def _note(counter, frame):
    site = _site(frame)
    with _lock:
        setattr(census, counter, getattr(census, counter) + 1)
        census.sites[site] = census.sites.get(site, 0) + 1
    return site


def _pauses():
    pauses = getattr(_thread, "pauses", None)
    if pauses is None:
        pauses = _thread.pauses = []
    return pauses


def _is_cuda(device):
    if not hasattr(device, "is_cuda"):
        device = wp.get_device(device)
    return bool(device.is_cuda)


def _on_cuda(scope):
    """Whether a Warp capture scope records on a CUDA device (its stream's, else its device argument)."""
    stream = getattr(scope, "stream", None)
    return _is_cuda(getattr(scope, "device", None) if stream is None else stream.device)


def _outside_captures():
    """Whether this thread has no package capture open and no CUDA device in use is capturing.

    A Krylov solve of stock Warp that is issued then captures its own loop into a single-use
    graph (warp/_src/optim/linear.py, ``_run_capturable_loop``). A CUDA device without a context
    holds no array and no capture; a process that uses none counts nothing.
    """
    if getattr(_thread, "captures", 0):
        return False
    used = False
    for device in wp.get_cuda_devices():
        if device.has_context:
            if device.is_capturing:
                return False
            used = True
    return used


def _enter_gc_scope(collect):
    global _depth, _restore_gc
    with _lock:
        if collect and _depth:
            raise RuntimeError("Explicit collection is only allowed outside capture scopes")
        if not _depth:
            if collect:
                gc.collect()
            _restore_gc = gc.isenabled()
            if _restore_gc:
                gc.disable()
        _depth += 1


def _leave_gc_scope():
    global _depth, _restore_gc
    with _lock:
        _depth -= 1
        if not _depth:
            if _restore_gc:
                gc.enable()
            _restore_gc = False


class PauseGC:
    """Delay cyclic cleanup around a call that may capture internally.

    This does not start a CUDA capture. Warp's Krylov solvers can create their
    own conditional graph even when the caller is outside an explicit capture.
    Nested scopes share the same lifetime accounting as :class:`ScopedCapture`.

    A pause that ends without an error and held no other pause is one Krylov call. It is
    counted as a solver capture when it was entered outside every capture (see
    ``_outside_captures``): that is the call for which stock Warp captures a graph of its own.

    The count follows this rule about pauses; it is not a count of Warp's captures. The two
    agree where each such pause holds exactly one Krylov call that captures
    (``use_cuda_graph=True`` on a CUDA device), which is how this package issues its solves.
    Measured against a count at ``wp.capture_begin``: a pause around a solve with
    ``use_cuda_graph=False``, around a CPU solve while a CUDA device is in use, or around no
    solve counts one where Warp captures none; a pause with two solves counts one for two; a
    pause that raises after its solve counts none for one. A Krylov solve outside every pause
    and a stock ``wp.ScopedCapture`` are not counted at all. A check for "no capture"
    therefore also counts at ``wp.capture_begin`` (``_counted_captures`` in
    tests/test_grasp_scene.py).
    """

    def __init__(self):
        self.entered = False

    def __enter__(self):
        if self.entered:
            raise RuntimeError("A GC pause cannot be entered twice")
        outside = _outside_captures()
        _enter_gc_scope(False)
        self.entered = True
        pauses = _pauses()
        if pauses:
            # The enclosing pause holds this one: it is not a single Krylov call.
            pauses[-1][0] = False
        pauses.append([True, outside])
        return self

    def __exit__(self, kind, *args):
        if not self.entered:
            raise RuntimeError("GC pause was not entered")
        self.entered = False
        try:
            single, outside = _pauses().pop()
            if single and outside and kind is None:
                _note("solver_captures", sys._getframe(1))
        finally:
            _leave_gc_scope()


class ScopedCapture:
    """Warp capture with cyclic GC paused until all overlapping scopes end.

    ``collect=True`` optionally collects stale cycles before entering the first
    scope. Runtime graph creation uses the default False, avoiding a collection
    pause. Nested and overlapping calls preserve the original GC setting.
    Unrelated code must not enable GC or explicitly collect during this scope;
    explicit reference-count destruction and foreign CUDA threads are outside
    this guard's contract. No synchronization or collection occurs by default.

    A scope on a CUDA device is counted in the census once Warp's scope is entered, with the
    place it was entered from; its graph is remembered weakly when the scope ends.
    """

    def __init__(self, *args, collect=False, **kwargs):
        self.capture = wp.ScopedCapture(*args, **kwargs)
        self.collect = bool(collect)
        self.entered = False
        self.site = None

    def __enter__(self):
        if self.entered:
            raise RuntimeError("A capture scope cannot be entered twice")
        cuda = _on_cuda(self.capture)
        _enter_gc_scope(self.collect)
        self.entered = True
        try:
            result = self.capture.__enter__()
        except BaseException:
            self.entered = False
            _leave_gc_scope()
            raise
        _thread.captures = getattr(_thread, "captures", 0) + 1
        self.site = _note("captures", sys._getframe(1)) if cuda else None
        return result

    def __exit__(self, *args):
        if not self.entered:
            raise RuntimeError("Capture scope was not entered")
        try:
            return self.capture.__exit__(*args)
        finally:
            self.entered = False
            _thread.captures = getattr(_thread, "captures", 1) - 1
            try:
                graph = None if self.site is None else getattr(self.capture, "graph", None)
                if graph is not None:
                    with _lock:
                        census.graphs[graph] = self.site
            finally:
                _leave_gc_scope()


def note_release(count=1):
    """Record that an owner dropped ``count`` captured graphs: a changed key or an evicted cache entry."""
    with _lock:
        census.releases += int(count)


def loaded_modules():
    """Kernel module builds loaded in this process: {(module name, device alias, block size): module hash}.

    Read from Warp's module registry; nothing is loaded, hashed or changed. A kernel that gains
    an overload gives its module another hash, so the build shows as loaded again.
    """
    result = {}
    for name, module in tuple(_warp_context.user_modules.items()):
        for executable in tuple(module.execs.values()):
            result[(name, executable.device.alias, executable.block_dim)] = executable.module_hash
    return result


def _snapshot():
    modules = loaded_modules()
    mark = Mark()
    with _lock:
        mark.time = time.perf_counter()
        mark.captures, mark.solver_captures, mark.releases = census.captures, census.solver_captures, census.releases
        mark.sites = dict(census.sites)
    mark.modules = modules
    mark.startup = None
    return mark


def _loads(before, after):
    """Module builds of ``after`` that ``before`` does not hold, or holds with another hash."""
    return sum(before.get(key) != digest for key, digest in after.items())


def begin_startup():
    """Open a start-up: the counters, the loaded module builds and the time of this call."""
    return _snapshot()


def finish_startup(begun, device):
    """End the start-up that ``begun`` opened: the mark that later captures and loads are counted from.

    Returns None on a CPU device, where no CUDA graph is captured. May be called again with
    the same ``begun``: the later mark replaces the earlier one, and its start-up record covers
    the time from ``begun`` to the later call.
    """
    if not _is_cuda(device):
        return None
    mark = _snapshot()
    with _lock:
        graphs = list(census.graphs)
    record = (
        mark.time - begun.time,
        mark.captures - begun.captures,
        mark.solver_captures - begun.solver_captures,
        _loads(begun.modules, mark.modules),
        # Warp builds the executable graph at the first launch: a graph without one was never launched.
        sum(graph.graph_exec is None for graph in graphs),
    )
    mark.startup = dict(zip(STARTUP_RECORD, record, strict=True))
    return mark


def counts(mark):
    """Captures since ``mark``, from the integer counters alone: nothing is walked."""
    return {
        "captures_after_startup": census.captures - mark.captures,
        "solver_captures_after_startup": census.solver_captures - mark.solver_captures,
    }


def _table(entries):
    """The ``TABLE_NAMES`` largest entries of {name: count}; the others are summed under "other"."""
    ordered = sorted(entries.items(), key=lambda item: (-item[1], item[0]))
    table = dict(ordered[:TABLE_NAMES])
    if len(ordered) > TABLE_NAMES:
        table["other"] = sum(count for _, count in ordered[TABLE_NAMES:])
    return table


def modules(mark, render_prefixes=()):
    """What a walk shows since ``mark``: module builds loaded, capture sites used, graphs released.

    Walks the loaded module builds once. A build that is new or has another hash than at the
    mark is one load; builds of modules whose name starts with one of ``render_prefixes`` are
    counted apart from the others. The tables name modules with their number of builds and
    capture sites with their number of captures (package captures and solver captures).
    """
    prefixes = tuple(render_prefixes)
    late, render = {}, {}
    for key, digest in loaded_modules().items():
        if mark.modules.get(key) != digest:
            table = render if key[0].startswith(prefixes) else late
            table[key[0]] = table.get(key[0], 0) + 1
    with _lock:
        sites = {site: count - mark.sites.get(site, 0) for site, count in census.sites.items()}
        released = census.releases - mark.releases
    return {
        "module_loads_after_startup": sum(late.values()),
        "render_module_loads_after_startup": sum(render.values()),
        "late_modules": _table(late),
        "late_render_modules": _table(render),
        "late_sites": _table({site: count for site, count in sites.items() if count}),
        "graphs_released_after_startup": released,
    }
