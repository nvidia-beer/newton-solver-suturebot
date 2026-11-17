# SPDX-License-Identifier: Apache-2.0
"""Issue a recorded run of kernel launches and memory operations as few launches.

Every kernel launch, memory fill and memory copy is a node with a fixed cost in the CUDA
graph that replays it, about one microsecond, whatever work it does. A run of small
launches therefore costs more in nodes than in work. ``record`` lists the operations that
a callable issues, in the order in which the device executes them, and ``FusedLaunches``
issues that list again with consecutive operations as stages of one generated kernel.

A generated kernel is launched with one CUDA block per world. A stage is the body of a
recorded kernel as a function that takes the thread index as arguments: the threads of
the block stride over the items of the launch it stands for and call that function, and a
block barrier ends the stage, so that a stage reads what the stages before it wrote. The
function is made from the source of the recorded kernel, in the scope of that kernel, so
a stage runs the arithmetic of the launch it stands for. A memory fill or copy is a stage
that writes the same bytes, one byte per item: a byte may alias an element of any type, so
the stages that use the memory with its own element type read what was written. On the CPU
a block is one thread, which runs the stages in their order.

What a generated kernel does not change: the operations and their order, the values of
every argument at the time of the recording, and the options of the Warp modules that the
stages are compiled with. What it changes: the threads of a stage are those of one block
instead of a grid of blocks, so the order in which the threads of one launch perform
atomic operations differs, as it does between two runs of that launch.

Operations that stay as they were recorded, each as a node of its own:

- a kernel that works across the threads of a block: it calls a tile function of Warp,
  ``wp.block_dim()`` or a function with native code;
- a kernel whose thread index is not a single ``wp.tid()`` assigned to names at the top
  level of the kernel body;
- a kernel with launch options (launch bounds, register limits, clusters) or in a module
  with deterministic atomics;
- a launch with more than ``STAGE_ITEMS_MAX`` items per block and a memory operation of
  more than ``MEMORY_BYTES_MAX`` bytes: the threads of one block would take more strides
  over them than a node of the graph costs;
- with more than one world, a launch that the world index does not lead (its first launch
  dimension is not the number of worlds or its first thread index is not named
  ``worldid``) and every memory operation, because they are not the work of one world.

With more than one world the blocks of a generated kernel run their stages independently
of each other. A launch led by the world index must therefore read and write only memory
of its world; the caller vouches for that by passing the number of worlds.
"""

import ast
import ctypes
import gc
import hashlib
import math
from dataclasses import dataclass

import warp as wp

from .capture import ScopedCapture
from .graph_nodes import KernelNode, MemcpyNode, MemsetNode, graph_nodes

# Most threads of a block: the block of a default Warp launch.
BLOCK_DIM_MAX = 256
# Most items that one block works through in a stage: two strides of a full block. A launch
# of its own spreads more items over more blocks, which gains more than its node costs.
STAGE_ITEMS_MAX = 2 * BLOCK_DIM_MAX
# Most bytes of a memory operation that is a stage: eight strides of a full block, a byte
# each, which take less time than the node of a copy.
MEMORY_BYTES_MAX = 8 * BLOCK_DIM_MAX
# Threads of one CUDA warp; a block is made of whole warps.
_WARP = 32
# Name of the leading thread index of a launch that runs per world.
_WORLD_INDEX = "worldid"
# Bytes of kernel arguments that a CUDA launch accepts (CUDA 12.1 and later), less the launch bounds.
_ARGUMENT_BYTES_MAX = 32764 - 64
# Upper bound of the bytes of an array argument that is passed as None.
_NULL_ARRAY_BYTES = 128
# Options of a recorded kernel that do not keep it from being a stage: they select how the
# kernel is launched or differentiated, not what its body computes.
_STAGE_KERNEL_OPTIONS = frozenset(("enable_backward", "grid_stride"))
# Module options that a generated kernel sets itself instead of taking them from its stages:
# they select how a kernel is launched, differentiated and named, not what a stage computes.
_OWN_MODULE_OPTIONS = frozenset(
    (
        "enable_backward",
        "block_dim",
        "default_grid_stride",
        "strip_hash",
        "deterministic",
        "deterministic_max_records",
    )
)
# Element types of the views that address a memory fill, by bytes per element.
_UNSIGNED = {1: wp.uint8, 2: wp.uint16, 4: wp.uint32, 8: wp.uint64}
# Type of the views that address memory byte by byte.
_BYTES = wp.array(dtype=wp.uint8, ndim=1)
# Symbol of the built-in Warp kernel that fills 2, 4 and 8 byte elements (warp/native/warp.cu).
_FILL_KERNEL = "memtile_value_kernel"

# Generated kernels by name. The name holds a digest of the source and of the kernels of
# the stages, so equal groups of one process share a kernel and a Warp module.
_kernels = {}


@wp.func_native(snippet="WP_TILE_SYNC();")
def _block_barrier(): ...


@dataclass(frozen=True, eq=False)
class Launch:
    """One kernel launch.

    Attributes:
        kernel: The launched ``wp.Kernel``.
        dim: Launch dimensions.
        inputs: Input arguments of the launch.
        outputs: Output arguments of the launch.
        block_dim: Threads per block of the launch; 0 selects the default of the device.
        max_blocks: Upper bound of the blocks of the launch; 0 for none.
    """

    kernel: wp.Kernel
    dim: tuple[int, ...]
    inputs: tuple = ()
    outputs: tuple = ()
    block_dim: int = 0
    max_blocks: int = 0


@dataclass(frozen=True, eq=False)
class Memset:
    """One memory fill: ``count`` elements of ``width`` bytes at ``pointer`` take ``value``.

    Attributes:
        pointer: Address of the first element on the device.
        value: The element as an unsigned integer.
        width: Bytes per element: 1, 2, 4 or 8.
        count: Number of elements.
        owners: Arrays that keep the memory alive.
    """

    pointer: int
    value: int
    width: int
    count: int
    owners: tuple = ()


@dataclass(frozen=True, eq=False)
class Memcpy:
    """One memory copy of ``size`` bytes from ``source`` to ``destination`` on one device.

    Attributes:
        destination: Address of the first byte written.
        source: Address of the first byte read.
        size: Number of bytes.
        owners: Arrays that keep the memory of both ranges alive.
    """

    destination: int
    source: int
    size: int
    owners: tuple = ()


def record(body, device):
    """List the kernel launches and memory operations that ``body`` issues on a CUDA device.

    ``body`` is called once inside a CUDA graph capture with a tape active. The capture
    executes nothing, so the arrays keep their contents. The tape gives the kernel and the
    arguments of every launch and the captured graph gives the order of all operations and
    the memory operations, which no tape records.

    ``body`` must be capturable in a graph without memory allocations: create its
    temporary arrays from a pool such as ``StepScratch`` that has served the same calls
    before. It must issue every launch on ``device`` with tape recording enabled, and it
    must not capture conditionals.

    Args:
        body: Callable without arguments.
        device: The CUDA device that ``body`` issues its operations on.

    Returns:
        A list of ``Launch``, ``Memset`` and ``Memcpy`` in execution order. Launches of size
        zero issue nothing and are not listed.

    Raises:
        RuntimeError: If ``device`` is not a CUDA device (there is no graph to read memory
            operations from; issue the operations of ``body`` unfused), if the graph holds
            a node that is neither a kernel launch nor a memory fill or copy, if its kernel
            nodes are not the launches of the tape, if a memory operation addresses memory
            that no array on the device holds, or if the device is capturing a graph.
    """
    device = wp.get_device(device)
    if not device.is_cuda:
        raise RuntimeError(
            f"record() reads memory operations from a CUDA graph capture and cannot run on device '{device}'; "
            "issue the operations unfused"
        )
    if device.is_capturing:
        raise RuntimeError("record() captures a graph of its own and cannot run during graph capture")
    with wp.Tape() as tape, ScopedCapture(device=device) as capture:
        body()
    launches = []
    for entry in tape.launches:
        if not isinstance(entry, list):
            continue
        kernel, dim, max_blocks, inputs, outputs, launch_device, block_dim, _metadata = entry
        if launch_device != device:
            raise RuntimeError(f"Launch of kernel {kernel.key} on device '{launch_device}' instead of '{device}'")
        dim = (dim,) if isinstance(dim, int) else tuple(dim)
        if math.prod(dim) > 0:
            launches.append(Launch(kernel, dim, tuple(inputs), tuple(outputs), block_dim, max_blocks))
    launches.reverse()

    owners = _Owners(device)
    operations = []
    for node in graph_nodes(capture.graph):
        if isinstance(node, MemsetNode):
            operations.append(
                Memset(
                    node.pointer, node.value, node.width, node.count, owners.of(node.pointer, node.width * node.count)
                )
            )
        elif isinstance(node, MemcpyNode):
            holders = owners.of(node.destination, node.size) + owners.of(node.source, node.size)
            operations.append(Memcpy(node.destination, node.source, node.size, holders))
        elif launches and node.function == _forward_function(launches[-1], device):
            operations.append(launches.pop())
        else:
            operations.append(_fill(node, owners))
    if launches:
        names = ", ".join(launch.kernel.key for launch in reversed(launches))
        raise RuntimeError(f"The captured graph holds no kernel node for the recorded launches of {names}")
    return operations


def _forward_function(launch, device):
    """Handle of the CUDA function that a launch runs."""
    executable = launch.kernel.module.load(device, launch.block_dim)
    return executable.get_kernel_hooks(launch.kernel).forward


def _fill(node: KernelNode, owners):
    """The memory fill that a kernel node of Warp's built-in fill kernel stands for."""
    if _FILL_KERNEL not in node.name or len(node.arguments) != 3:
        raise RuntimeError(
            f"The captured graph launches {node.name}, which is not the next launch of the tape: a launch without "
            "tape recording, or a fill or copy that Warp issues as a built-in kernel other than "
            f"{_FILL_KERNEL} (a fill of elements wider than 8 bytes or of an array that is not contiguous)"
        )
    pointer, value, count = (int.from_bytes(argument, "little") for argument in node.arguments)
    width = len(node.arguments[1])
    return Memset(pointer, value, width, count, owners.of(pointer, width * count))


class _Owners:
    """The arrays of a device that hold given memory ranges."""

    def __init__(self, device):
        self._device = device
        self._arrays = None

    def of(self, pointer, size):
        """Arrays that hold the ``size`` bytes at ``pointer``; raises if there is none."""
        if self._arrays is None:
            # Warp keeps no registry of arrays; the collector knows every living one.
            self._arrays = [
                array
                for array in gc.get_objects()
                if isinstance(array, wp.array) and getattr(array, "ptr", None) and array.device == self._device
            ]
        holders = tuple(
            array for array in self._arrays if array.ptr <= pointer and pointer + size <= array.ptr + array.capacity
        )
        if not holders:
            raise RuntimeError(
                f"A memory operation addresses {size} bytes at {pointer:#x}, which no array on '{self._device}' holds"
            )
        return holders


def _scope(function):
    """Names that the body of a Python function resolves outside itself: globals, then closure."""
    scope = dict(function.__globals__)
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True):
        try:
            scope[name] = cell.cell_contents
        except ValueError:  # a cell that nothing is bound to yet
            pass
    return scope


def _path(node):
    """The names of a dotted reference, or None for another expression."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    names.append(node.id)
    return names[::-1]


def _definition(adjoint):
    """A copy of the function definition that an adjoint of Warp was made from, or None for other source."""
    definition = ast.parse(adjoint.source).body[0]
    return definition if isinstance(definition, ast.FunctionDef) else None


def _local_names(definition):
    arguments = definition.args
    names = {argument.arg for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    names.update(
        node.id for node in ast.walk(definition) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    return names


def _names_tid(node):
    """Whether an expression names something called ``tid``."""
    return (isinstance(node, ast.Attribute) and node.attr == "tid") or (isinstance(node, ast.Name) and node.id == "tid")


def _is_block_builtin(name):
    """Whether a Warp builtin of this name works across the threads of a block."""
    return name.startswith("tile") or name in ("untile", "block_dim")


def _block_cooperation(adjoint, seen):
    """Name of what makes the code of an adjoint work across the threads of a block, or None.

    Follows the Warp functions that the code references, by name or as the value of a
    static expression. A name that could be a tile builtin counts as one.
    """
    definition = _definition(adjoint)
    if definition is None:
        return f"{adjoint.fun_name} is not defined as a function"
    scope, local = _scope(adjoint.func), _local_names(definition)
    functions = [value for _code, value in adjoint.resolved_static_expressions.items()]
    functions.extend(value for _code, value in adjoint.deferred_static_expressions)
    for node in ast.walk(definition):
        path = _path(node) if isinstance(node, (ast.Name, ast.Attribute)) else None
        if path is None or path[0] in local:
            continue
        if _is_block_builtin(path[-1]):
            return path[-1]
        value = scope.get(path[0])
        for name in path[1:]:
            value = getattr(value, name, None)
        functions.append(value)
    for function in functions:
        if not isinstance(function, wp.Function) or function.is_builtin():
            continue
        for overload in (function, *function.user_overloads.values(), *function.user_templates.values()):
            if id(overload) in seen:
                continue
            seen.add(id(overload))
            if overload.native_snippet is not None:
                return f"native function {overload.key}"
            found = _block_cooperation(overload.adj, seen)
            if found is not None:
                return found
    return None


@dataclass(frozen=True, eq=False)
class _Body:
    """The body of a kernel as a function that takes the thread index as leading arguments.

    Attributes:
        name: Name of the function; it ends with the content hash of the kernel.
        source: Source of the function.
        indices: Names that the kernel assigns ``wp.tid()`` to; they lead the arguments.
        function: The Python function that ``source`` defines in the scope of the kernel.
    """

    name: str
    source: str
    indices: tuple[str, ...]
    function: object


def _body(kernel):
    """The ``_Body`` of a kernel, or the reason why the kernel cannot be a stage."""
    if kernel.hash is None:
        # hashes the kernels of the module, which also resolves their static expressions
        kernel.module.get_module_hash()
    if kernel.hash is None:
        raise RuntimeError(f"Kernel {kernel.key} has no content hash; its module {kernel.module.name} cannot be built")
    cooperation = _block_cooperation(kernel.adj, set())
    if cooperation is not None:
        return f"works across the threads of a block ({cooperation})"
    definition = _definition(kernel.adj)
    arguments = definition.args
    if arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg or arguments.kwarg or arguments.defaults:
        return "has arguments other than plain positional ones"

    calls = [node for node in ast.walk(definition) if isinstance(node, ast.Call) and _names_tid(node.func)]
    indices = ()
    if calls:
        statements = [
            statement
            for statement in definition.body
            if isinstance(statement, ast.Assign) and statement.value is calls[0] and len(statement.targets) == 1
        ]
        path = _path(calls[0].func)
        plain = path is not None and len(path) == 2 and _scope(kernel.func).get(path[0]) is wp
        if len(calls) > 1 or not statements or not plain:
            return "reads the thread index other than by one wp.tid() assigned at the top level of its body"
        target = statements[0].targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else [target]
        if not all(isinstance(name, ast.Name) for name in targets):
            return "assigns wp.tid() to something other than names"
        indices = tuple(name.id for name in targets)
        if len(set(indices)) != len(indices) or any(argument.arg in indices for argument in arguments.args):
            return "names a thread index like an argument or another thread index"
        definition.body[definition.body.index(statements[0])] = ast.Pass()

    name = f"stage_{kernel.key}_{kernel.hash.hex()[:16]}"
    definition.name = name
    definition.decorator_list = []
    definition.returns = None
    for argument in arguments.args:
        argument.annotation = None
    arguments.args[:0] = [ast.arg(arg=index) for index in indices]
    source = ast.unparse(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))) + "\n"
    scope = _scope(kernel.func)
    exec(compile(source, f"<{name}>", "exec"), scope)
    return _Body(name, source, indices, scope[name])


def _argument_bytes(value):
    """Upper bound of the bytes that a value takes among the arguments of a launch."""
    if value is None:
        return _NULL_ARRAY_BYTES
    if hasattr(value, "__ctype__"):
        value = value.__ctype__()
    try:
        size = ctypes.sizeof(value)
    except TypeError:  # Python and Warp scalars
        size = 8
    return 8 * math.ceil(size / 8)


def _coordinates(dim, arity, per_world):
    """Source of the coordinates that ``wp.tid()`` unpacks into ``arity`` names for one item of a launch.

    ``item`` counts the threads of the launch, or with ``per_world`` those of the world
    ``world``. Warp pads a launch that has fewer dimensions than names with ones and gives
    the threads of further dimensions the same coordinates (``launch_coord`` in
    warp/native/builtin.h).
    """
    kept = (dim + (1,) * arity)[:arity]
    shared = math.prod(dim[arity:])
    first = 1 if per_world else 0
    coordinates = ["world"] * first
    for axis in range(first, arity):
        stride = shared * math.prod(kept[axis + 1 :])
        coordinate = "item" if stride == 1 else f"item // {stride}"
        if kept[axis] == 1:
            coordinate = "0"
        elif math.prod(kept[first:axis]) > 1:  # else the items end before the coordinate wraps
            coordinate = f"({coordinate}) % {kept[axis]}"
        coordinates.append(coordinate)
    return coordinates


class _Group:
    """Consecutive operations that one generated kernel issues as stages."""

    def __init__(self, nworld):
        self.nworld = nworld
        self.operations = []
        self.labels = []
        self.options = None
        self.bytes = 0
        self.width = 1
        self._lines = []
        self._bodies = {}
        self._annotations = []
        self._values = []
        self._arrays = {}

    def accepts(self, values, options):
        """Whether a stage with these argument values and module options can join the group."""
        if options is not None and self.options is not None and options != self.options:
            return False
        return self.bytes + sum(_argument_bytes(value) for value in values) <= _ARGUMENT_BYTES_MAX

    def _argument(self, annotation, value):
        """Name of the kernel argument that passes ``value``; an array that is passed twice takes one argument."""
        key = None
        if isinstance(value, wp.array):
            key = (value.ptr, value.shape, value.strides, wp.types.type_repr(annotation))
            if key in self._arrays:
                return self._arrays[key]
        name = f"a{len(self._values)}"
        self._annotations.append(annotation)
        self._values.append(value)
        self.bytes += _argument_bytes(value)
        if key is not None:
            self._arrays[key] = name
        return name

    def _stage(self, operation, label, items, call):
        self.operations.append(operation)
        self.labels.append(label)
        self.width = max(self.width, items)
        self._lines.append(f"    # {label}")
        self._lines.append(f"    for item in range(thread, {items}, width):")
        self._lines.append(f"        {call}")
        self._lines.append("    _block_barrier()")

    def add_launch(self, launch, kernel, body, options):
        self.options = options
        self._bodies[body.name] = (body, kernel)
        values = (*launch.inputs, *launch.outputs)
        names = [self._argument(variable.type, value) for variable, value in zip(kernel.adj.args, values, strict=True)]
        per_world = self.nworld > 1
        coordinates = _coordinates(launch.dim, len(body.indices), per_world)
        items = math.prod(launch.dim) // self.nworld if per_world else math.prod(launch.dim)
        label = f"{kernel.key} {launch.dim}"
        self._stage(launch, label, items, f"{body.name}({', '.join(coordinates + names)})")

    def add_memset(self, memset, view):
        target = self._argument(_BYTES, view)
        element = memset.value.to_bytes(memset.width, "little")
        if len(set(element)) == 1:
            fill = self._argument(wp.uint8, wp.uint8(element[0]))
            store = f"{target}[item] = {fill}"
        else:
            # the byte of its element that the item is
            fill = self._argument(wp.uint64, wp.uint64(memset.value))
            store = f"{target}[item] = wp.uint8({fill} >> wp.uint64(8 * (item % {memset.width})))"
        self._stage(memset, f"memset {view.size} bytes", view.size, store)

    def add_memcpy(self, memcpy, destination, source):
        target, origin = self._argument(_BYTES, destination), self._argument(_BYTES, source)
        self._stage(memcpy, f"memcpy {memcpy.size} bytes", memcpy.size, f"{target}[item] = {origin}[item]")

    def kernel(self):
        """The generated kernel of the group, its argument values and its threads per block."""
        names = [f"a{index}" for index in range(len(self._values))]
        # the barrier after the last stage separates nothing
        lines = ["    world, thread = wp.tid()", "    width = wp.block_dim()", *self._lines[:-1]]
        options = dict(self.options or {})
        digest = hashlib.sha256("\n".join(lines).encode())
        for annotation in self._annotations:
            digest.update(wp.types.type_repr(annotation).encode())
        digest.update(repr(sorted((key, str(value)) for key, value in options.items())).encode())
        name = f"fused_{digest.hexdigest()[:16]}"
        if name not in _kernels:
            module = wp.get_module(f"fused_launches.{name}")
            module.options.update(options, enable_backward=False)
            scope = {"wp": wp, "_block_barrier": _block_barrier}
            for body, kernel in self._bodies.values():
                annotations = dict.fromkeys(body.indices, int)
                annotations.update((variable.label, variable.type) for variable in kernel.adj.args)
                scope[body.name] = wp.Function(
                    func=body.function,
                    key=body.name,
                    namespace="",
                    module=module,
                    source=body.source,
                    overloaded_annotations=annotations,
                    scope_locals={},
                )
            source = f"def {name}({', '.join(names)}):\n" + "\n".join(lines) + "\n"
            exec(compile(source, f"<{name}>", "exec"), scope)
            function = scope[name]
            function.__annotations__ = dict(zip(names, self._annotations, strict=True))
            _kernels[name] = wp.Kernel(
                func=function, key=name, module=module, options={"enable_backward": False}, source=source
            )
        block_dim = min(BLOCK_DIM_MAX, _WARP * math.ceil(self.width / _WARP))
        return _kernels[name], list(self._values), block_dim


class FusedLaunches:
    """A recorded sequence of kernel launches and memory operations, issued with fewer launches.

    Consecutive operations that can be stages of one kernel are issued as one launch of a
    generated kernel with one block per world; the module docstring says which operations
    can. Every other operation is issued as it was recorded. ``launch()`` leaves in the
    arrays what issuing the recorded operations one after the other leaves there, up to
    the order of the atomic operations of one launch.

    The arguments keep the values of the recording: an array is the same memory, a scalar
    the same number. Record again after a change of either. The object keeps every array
    that its operations address alive. A captured graph addresses the kernels of this
    object, so keep the object alive while such a graph is launched.

    Creating the object generates the kernels; Warp compiles a generated kernel at its
    first launch and loads it from the kernel cache in later processes. ``launch()``
    allocates nothing and reads nothing back, so it can be captured in a graph and in the
    body of a graph conditional. The generated kernels have no adjoint.

    Args:
        operations: ``Launch``, ``Memset`` and ``Memcpy`` in execution order, as ``record``
            returns them.
        nworld: Number of worlds: the blocks of a generated kernel. With more than one
            world, a launch led by the world index must read and write only memory of its
            world.
        device: Device of the operations.

    Attributes:
        node_count: Number of graph nodes that ``launch()`` issues.

    Raises:
        ValueError: If an operation is none of the three records, if a launch does not
            pass one value per kernel argument, or if a memory fill has elements of another
            width than 1, 2, 4 or 8 bytes.
    """

    def __init__(self, operations, nworld, device):
        self.device = wp.get_device(device)
        self.nworld = int(nworld)
        if self.nworld < 1:
            raise ValueError(f"FusedLaunches needs at least one world, got {nworld}")
        self._operations = tuple(operations)
        self._bodies = {}
        self._issues = []
        self._lines = []
        group = _Group(self.nworld)
        for operation in self._operations:
            if isinstance(operation, Launch):
                stage = self._launch_stage(operation)
            elif isinstance(operation, Memset):
                stage = self._memset_stage(operation)
            elif isinstance(operation, Memcpy):
                stage = self._memcpy_stage(operation)
            else:
                raise ValueError(f"FusedLaunches takes Launch, Memset and Memcpy, got {type(operation).__name__}")
            if stage is None:  # a launch without threads issues nothing
                continue
            add, values, options = stage
            if isinstance(add, str) or not group.accepts(values, options):
                self._close(group)
                group = _Group(self.nworld)
            if isinstance(add, str):
                self._unfused(operation, add)
            else:
                add(group)
        self._close(group)
        self.node_count = len(self._issues)

    def launch(self):
        """Issue the sequence on the device."""
        for issue in self._issues:
            issue()

    def describe(self):
        """One line per graph node that ``launch()`` issues: what it runs and, if it is not fused, why."""
        return list(self._lines)

    def _launch_stage(self, launch):
        """How a launch joins a group: (function that adds it, or why it cannot, its values, its module options)."""
        kernel, dim = launch.kernel, tuple(launch.dim)
        values = (*launch.inputs, *launch.outputs)
        if len(values) != len(kernel.adj.args):
            raise ValueError(f"Launch of {kernel.key} with {len(values)} values for {len(kernel.adj.args)} arguments")
        if math.prod(dim) == 0:
            return None
        if kernel.is_generic:
            # the overload that the launch instantiates, as wp.launch selects it
            kernel = kernel.add_overload(kernel.infer_argument_types(list(values)))
        if kernel not in self._bodies:
            self._bodies[kernel] = _body(kernel)
        body = self._bodies[kernel]
        module_options = kernel.module.options
        options = {key: value for key, value in module_options.items() if key not in _OWN_MODULE_OPTIONS}
        items = math.prod(dim) // self.nworld if self.nworld > 1 else math.prod(dim)
        if isinstance(body, str):
            return body, values, options
        if not set(kernel.options) <= _STAGE_KERNEL_OPTIONS:
            return "has launch options", values, options
        if module_options["deterministic"] != wp.DeterministicMode.NOT_GUARANTEED:
            return "is in a module with deterministic atomics", values, options
        if self.nworld > 1 and (dim[0] != self.nworld or body.indices[:1] != (_WORLD_INDEX,)):
            return "is not led by the world index", values, options
        if items > STAGE_ITEMS_MAX:
            return f"has {items} items per block", values, options
        return (lambda group: group.add_launch(launch, kernel, body, options)), values, options

    def _memset_stage(self, memset):
        """How a memory fill joins a group, as ``_launch_stage``."""
        if memset.width not in _UNSIGNED:
            raise ValueError(f"A memory fill with elements of {memset.width} bytes is not supported")
        size = memset.width * memset.count
        reason = self._memory_reason(size)
        if reason:
            return reason, (), None
        view = self._view(memset.pointer, 1, size)
        return (lambda group: group.add_memset(memset, view)), (view, memset.value), None

    def _memcpy_stage(self, memcpy):
        """How a memory copy joins a group, as ``_launch_stage``."""
        reason = self._memory_reason(memcpy.size)
        if reason:
            return reason, (), None
        destination = self._view(memcpy.destination, 1, memcpy.size)
        source = self._view(memcpy.source, 1, memcpy.size)
        return (lambda group: group.add_memcpy(memcpy, destination, source)), (destination, source), None

    def _memory_reason(self, size):
        """Why a memory operation of ``size`` bytes keeps a node of its own, or None."""
        if self.nworld > 1:
            return "is not the work of one world"
        if size > MEMORY_BYTES_MAX:
            return f"has {size} bytes"
        return None

    def _view(self, pointer, width, count):
        """Array of ``count`` unsigned elements of ``width`` bytes at an address of the device; it owns nothing."""
        return wp.array(ptr=pointer, dtype=_UNSIGNED[width], shape=(count,), device=self.device)

    def _close(self, group):
        """Issue the operations of a group: as one generated kernel if there are several."""
        if len(group.operations) == 1:
            self._unfused(group.operations[0], "is alone between operations that are not fused")
        elif group.operations:
            kernel, values, block_dim = group.kernel()
            nworld, device = self.nworld, self.device
            self._issues.append(
                lambda: wp.launch_tiled(kernel, dim=[nworld], inputs=values, block_dim=block_dim, device=device)
            )
            stages = "; ".join(group.labels)
            self._lines.append(
                f"{kernel.key}: {len(group.operations)} operations, {len(values)} arguments of {group.bytes} bytes, "
                f"block of {block_dim}: {stages}"
            )

    def _unfused(self, operation, reason):
        """Issue an operation as it was recorded."""
        device = self.device
        if isinstance(operation, Launch):
            self._issues.append(
                lambda: wp.launch(
                    operation.kernel,
                    dim=operation.dim,
                    inputs=operation.inputs,
                    outputs=operation.outputs,
                    device=device,
                    block_dim=operation.block_dim,
                    max_blocks=operation.max_blocks,
                )
            )
            label = f"{operation.kernel.key} {tuple(operation.dim)}"
        elif isinstance(operation, Memset):
            view = self._view(operation.pointer, operation.width, operation.count)
            self._issues.append(lambda: view.fill_(operation.value))
            label = f"memset {operation.width * operation.count} bytes"
        else:
            destination = self._view(operation.destination, 1, operation.size)
            source = self._view(operation.source, 1, operation.size)
            self._issues.append(lambda: wp.copy(destination, source))
            label = f"memcpy {operation.size} bytes"
        self._lines.append(f"{label}: {reason}")
