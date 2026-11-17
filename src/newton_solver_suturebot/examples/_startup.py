# SPDX-License-Identifier: Apache-2.0
"""Start-up of an example: its graphs captured and launched and its kernel modules loaded before the first frame.

An example builds its scene at construction and again at every rebuild (a reset, a region
switch, a recording switch). Each of these is a start-up. It opens with :meth:`Startup.begin`
and ends with :meth:`Startup.finish`, the mark, once every CUDA graph of the scene is captured
and has been launched, every kernel module the scene can need is loaded
(:func:`load_kernel_modules`) and the start state has been drawn (:func:`first_draw`). A graph
capture, a kernel module load or the first launch of a start-up graph after the mark falls
into a frame of the run or into a user action.

The self-test of an example (``--test``) plays a :class:`Script` of user actions between its
frames and ends with :meth:`Startup.check`, which fails on any of them.

The counters, the mark and the tables are those of ``_src/utils/capture.py``. They are kept
for the whole process: the check is meant for a process that runs one example.
"""

import weakref

import warp as wp

from newton_solver_suturebot._src.utils import capture


def owns_startup(example, cls):
    """Whether ``cls`` opens, marks and checks the start-up of ``example``: only as its most-derived class.

    A scene built on an example class has more to prepare than that class knows of, opens its
    start-up at its own point and marks it at its own end. The class asks this in its
    constructor and leaves an instance of a derived class alone.
    """
    return type(example) is cls


def frame_limited(viewer):
    """Whether the viewer ends the run by itself after a number of the frames it is given.

    The null and USD viewers count their frames up to ``num_frames``; the OpenGL and RTX
    viewers do so only without a window (a window runs until it is closed and ignores the
    limit). A viewer without ``num_frames`` has no limit.
    """
    if getattr(viewer, "num_frames", None) is None:
        return False
    return bool(getattr(viewer, "_headless", True))


# Kernel module of the log calls that every Newton viewer shares (shape transforms, lines, points).
VIEWER_KERNELS = "newton._src.viewer.kernels"


def records_logs(viewer):
    """Whether the viewer keeps every log call in a recording, as a sample at the time of its frame.

    The USD viewer writes a time sample for each logged attribute and the file viewer a record
    for each logged state. A log call that is issued outside the frames of the run adds a
    sample to that recording.
    """
    from newton.viewer import ViewerFile, ViewerUSD  # noqa: PLC0415

    return isinstance(viewer, (ViewerUSD, ViewerFile))


def first_draw(viewer, draw, time=0.0, *, frame=True):
    """Issue the launches of one drawn frame during a start-up, so that the first frame of the run loads nothing.

    A viewer that records what is logged (:func:`records_logs`) is not drawn to: the start
    state would become a sample of its recording, at the time code that the frame before it
    left. Its log calls launch kernels of one module (``VIEWER_KERNELS``), which is loaded by
    name instead.

    Args:
        viewer: Newton viewer.
        draw: Makes the log calls of the example's ``render()``, those between ``begin_frame``
            and ``end_frame``.
        time: Simulated time of the drawn state [s].
        frame: Draw a whole frame where the viewer has no frame limit. A frame-limited viewer
            (:func:`frame_limited`) counts a frame in ``begin_frame`` (USD) or in ``end_frame``
            (null, OpenGL without a window) and would simulate one frame less than it was asked
            for: for it the log calls are issued with no frame around them. ``False`` asks for
            that from every viewer: a rebuild that a panel button starts runs inside the
            viewer's ``end_frame``, where no other frame can begin.
    """
    if records_logs(viewer):
        device = wp.get_device(getattr(viewer, "device", None))
        wp.load_module(VIEWER_KERNELS, device=device, block_dim=256 if device.is_cuda else 1)
        return
    whole = frame and not frame_limited(viewer)
    if whole:
        viewer.begin_frame(time)
    draw()
    if whole:
        viewer.end_frame()


def load_kernel_modules(device, *kernels, block_dim=256):
    """Load the kernel module builds that the first launch of each kernel would load.

    A launch on a CUDA device loads the build of its block size, 256 unless the launch names
    another; a launch on the CPU loads the build of block size 1. The module is taken from the
    kernel, not from a name: a kernel of the file that runs as the example (``python -m``)
    belongs to the module ``__main__``.
    """
    device = wp.get_device(device)
    size = block_dim if device.is_cuda else 1
    modules = []
    for kernel in kernels:
        if not any(kernel.module is module for module in modules):
            modules.append(kernel.module)
            wp.load_module(kernel.module, device=device, block_dim=size)


def launch_startup_graph(graph, device):
    """Launch a graph that a start-up captured, once, and wait for it.

    Warp builds the executable of a captured graph at its first launch. The caller puts the
    state of the scene back afterwards: the launch runs the captured work on the start state.
    """
    wp.capture_launch(graph)
    wp.synchronize_device(device)


# The numbers of a census that add up over the start-ups of one example.
_LATE_COUNTS = (
    "captures_after_startup",
    "solver_captures_after_startup",
    "module_loads_after_startup",
    "render_module_loads_after_startup",
    "graphs_released_after_startup",
)
_LATE_TABLES = ("late_sites", "late_modules", "late_render_modules")


def census_failure(census):
    """What a census of one scene shows that must not be there, as text, or None.

    The scene is checked when its status is "checked". Then no CUDA graph was captured after a
    start-up mark, by the package scope or by a Krylov solve of stock Warp, no kernel module
    was loaded (those of ``render_prefixes`` are counted apart and do not fail), and every
    graph that a start-up captured was launched in it.
    """
    if census.get("capture_census_status") != "checked":
        return None
    numbers = {name: census.get(name) for name in Startup.numbers}
    if all(value == 0 for value in numbers.values()):
        return None
    return (
        f"{numbers}, late sites {census.get('late_sites')}, late kernel modules {census.get('late_modules')}, "
        f"in {census.get('startups')} start-ups"
    )


class Startup:
    """The start-ups of one scene, and what its process captured and loaded after each of them.

    Args:
        device: Device of the scene. On a CPU device no CUDA graph is captured: no mark is
            taken and nothing is checked.
        render_prefixes: Names that begin the kernel modules whose loads after a mark are
            reported apart from the others and never fail the check; none by default, so that
            every kernel module load after a mark fails it.
    """

    # What must be zero in a checked scene.
    numbers = (
        "captures_after_startup",
        "solver_captures_after_startup",
        "module_loads_after_startup",
        "startup_unlaunched_graphs",
    )

    def __init__(self, device, *, render_prefixes=()):
        self.device = device
        self.render_prefixes = tuple(render_prefixes)
        # "start-up not marked", "CPU device" or "checked".
        self.status = "start-up not marked"
        self.begun = self.mark = None
        # Graphs of the package scope that were alive when the open start-up began.
        self._known = None
        # Record of the open or last start-up, with the unlaunched graphs of that start-up alone.
        self._record = None
        # Start-ups that a later one has closed: their number, their records added up, and what
        # came after their marks.
        self.closed = 0
        self._closed_record = dict.fromkeys(capture.STARTUP_RECORD, 0)
        self._closed_late = {**dict.fromkeys(_LATE_COUNTS, 0), **{name: {} for name in _LATE_TABLES}}

    @property
    def startups(self):
        """Start-ups that have been marked: the construction of the scene and each of its rebuilds."""
        return self.closed + (self.mark is not None)

    def begin(self):
        """Open a start-up: everything from here to :meth:`finish` belongs to it.

        What the process captured and loaded after the mark of the previous start-up is kept:
        it fell into frames of the run or into the user action that asks for this rebuild.
        """
        self._close()
        self._known = weakref.WeakSet(tuple(capture.census.graphs))
        self.begun = capture.begin_startup()
        self.status = "start-up not marked"

    def finish(self):
        """End the open start-up here: the mark that later captures and loads are counted from.

        May be called again before the next :meth:`begin`: the later mark replaces the earlier
        one, and the record of the start-up covers the time to the later call. Returns the
        mark, or None on a CPU device.
        """
        if self.begun is None:
            raise RuntimeError("No start-up is open: call begin() first")
        mark = capture.finish_startup(self.begun, self.device)
        if mark is None:
            self.status = "CPU device"
            return None
        record = dict(mark.startup)
        # Of this start-up alone: a graph that another scene of the process left unlaunched is not one of ours.
        record["startup_unlaunched_graphs"] = sum(
            graph not in self._known and graph.graph_exec is None for graph in tuple(capture.census.graphs)
        )
        self.mark, self._record, self.status = mark, record, "checked"
        return mark

    def _late(self):
        """What came after the current mark: one walk of the loaded kernel modules."""
        late = capture.modules(self.mark, self.render_prefixes)
        return {**capture.counts(self.mark), **{name: late[name] for name in (*_LATE_COUNTS[2:], *_LATE_TABLES)}}

    def _close(self):
        """Add the marked start-up, and what came after its mark, to the totals of the closed ones."""
        if self.mark is None:
            return
        late = self._late()
        for name in _LATE_COUNTS:
            self._closed_late[name] += late[name]
        for name in _LATE_TABLES:
            table = self._closed_late[name]
            for key, count in late[name].items():
                table[key] = table.get(key, 0) + count
        for name in capture.STARTUP_RECORD:
            self._closed_record[name] += self._record[name]
        self.closed += 1
        self.mark = self._record = None

    def census(self):
        """The start-ups of this scene and what came after their marks; one walk of the loaded kernel modules.

        Returns ``capture_census_status`` and ``startups``. For a checked scene also: the
        captures, solver captures, kernel module loads and viewer kernel module loads after a
        mark, the record of the start-ups (seconds, captures, solver captures, kernel module
        loads and graphs never launched, each added up over the start-ups) and the tables of
        the late capture sites and late kernel modules.
        """
        result = {"capture_census_status": self.status, "startups": self.startups}
        if self.mark is None:
            return result
        late = self._late()
        for name in _LATE_COUNTS:
            result[name] = self._closed_late[name] + late[name]
        for name in capture.STARTUP_RECORD:
            result[name] = self._closed_record[name] + self._record[name]
        for name in _LATE_TABLES:
            table = dict(self._closed_late[name])
            for key, count in late[name].items():
                table[key] = table.get(key, 0) + count
            result[name] = table
        return result

    def check(self):
        """Fail if something was captured or loaded after a start-up mark, or a start-up graph was never launched.

        A scene without a mark is not checked, and its reason is printed (a CPU device).

        Returns:
            The census without its tables, for the line that ``test_final`` prints.
        """
        census = self.census()
        if census["capture_census_status"] != "checked":
            print(
                f"Captures and kernel module loads after start-up are not checked: {census['capture_census_status']}",
                flush=True,
            )
        failure = census_failure(census)
        if failure is not None:
            raise AssertionError(
                "Graph captures or kernel module loads after start-up, or a start-up graph that was never "
                f"launched: {failure}"
            )
        return {name: value for name, value in census.items() if name not in _LATE_TABLES}


class Script:
    """The scripted user of a self-test: actions between the frames of a run.

    The example's ``step()`` calls :meth:`play` before the physics of the frame; the actions of
    that frame run in the order they were added. They call what the handlers of the mouse and
    of the panel call.
    """

    def __init__(self):
        # Frames played so far, and frames the script needs: its last action and the rest after it.
        self.frame = 0
        self.frames = 0
        self._actions = {}
        # (frame, label) of every action that ran, in order; a drag is listed at its first frame.
        self.done = []

    def at(self, frame, label, action):
        """Call ``action()`` before the physics of ``frame``."""
        self._actions.setdefault(int(frame), []).append((label, action, True))
        self.frames = max(self.frames, int(frame) + 1)

    def drag(self, first, last, label, action):
        """Call ``action(progress)`` before every frame from ``first`` to ``last``.

        The progress rises in equal parts and is 1 at ``last``.
        """
        first, last = int(first), int(last)
        if last < first:
            raise ValueError("A drag ends at or after its first frame")
        count = last - first + 1
        for index in range(count):
            progress = (index + 1) / count
            self._actions.setdefault(first + index, []).append(
                (label, lambda progress=progress: action(progress), index == 0)
            )
        self.frames = max(self.frames, last + 1)

    def rest(self, frames):
        """Run ``frames`` more frames after the last action."""
        self.frames += int(frames)

    def reserve(self, viewer):
        """Let a frame-limited viewer run every frame of the script: a shorter limit is raised to its length."""
        if frame_limited(viewer) and viewer.num_frames < self.frames:
            viewer.num_frames = self.frames

    @property
    def complete(self):
        return self.frame >= self.frames

    def run(self, label, action, listed=True):
        """Run one action; a tool that counts what an action does wraps this call."""
        action()
        if listed:
            self.done.append((self.frame, label))

    def play(self):
        """Run the actions of the next frame, then count the frame."""
        for label, action, listed in self._actions.get(self.frame, ()):
            self.run(label, action, listed)
        self.frame += 1

    def summary(self):
        """For the line that ``test_final`` prints: frames played of the frames needed, and actions run."""
        return {"script_frames": [self.frame, self.frames], "script_actions": len(self.done)}
