# SPDX-License-Identifier: Apache-2.0
"""Read the nodes of a captured CUDA graph.

A CUDA graph that Warp captures from one stream is a chain of nodes: kernel launches,
memory fills and memory copies, each depending on the node before it. Warp records the
kernel launches on a tape but not the memory operations, so the graph is the only place
that holds the complete sequence a callable issues. ``graph_nodes`` returns that sequence.

The nodes are read with the CUDA driver functions ``cuGraphGetNodes``,
``cuGraphNodeGetType``, ``cuGraphNodeGetDependencies``, ``cuGraphKernelNodeGetParams_v2``,
``cuGraphMemsetNodeGetParams``, ``cuGraphMemcpyNodeGetParams``, ``cuFuncGetName`` and
``cuFuncGetParamInfo``, which Warp does not expose. Nothing is written to the graph.
"""

import ctypes
from dataclasses import dataclass

from .graph_branches import _load_driver

_KERNEL, _MEMCPY, _MEMSET = 0, 1, 2
_NODE_TYPES = {
    3: "host callback",
    4: "child graph",
    5: "empty",
    6: "event wait",
    7: "event record",
    8: "semaphore signal",
    9: "semaphore wait",
    10: "memory allocation",
    11: "memory free",
    12: "batched memory operation",
    13: "conditional",
}
_MEMORY_TYPE_DEVICE, _MEMORY_TYPE_UNIFIED = 2, 4
_prototyped = False


class _KernelParams(ctypes.Structure):
    """CUDA_KERNEL_NODE_PARAMS_v2 of cuda.h."""

    _fields_ = [
        ("func", ctypes.c_void_p),
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("kern", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
    ]


class _MemsetParams(ctypes.Structure):
    """CUDA_MEMSET_NODE_PARAMS of cuda.h."""

    _fields_ = [
        ("dst", ctypes.c_uint64),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("elementSize", ctypes.c_uint),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
    ]


class _MemcpyParams(ctypes.Structure):
    """CUDA_MEMCPY3D of cuda.h."""

    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t),
        ("srcY", ctypes.c_size_t),
        ("srcZ", ctypes.c_size_t),
        ("srcLOD", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_uint),
        ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_uint64),
        ("srcArray", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t),
        ("srcHeight", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t),
        ("dstY", ctypes.c_size_t),
        ("dstZ", ctypes.c_size_t),
        ("dstLOD", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_uint),
        ("dstHost", ctypes.c_void_p),
        ("dstDevice", ctypes.c_uint64),
        ("dstArray", ctypes.c_void_p),
        ("reserved1", ctypes.c_void_p),
        ("dstPitch", ctypes.c_size_t),
        ("dstHeight", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t),
        ("Height", ctypes.c_size_t),
        ("Depth", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class KernelNode:
    """One kernel launch of a graph.

    Attributes:
        function: Handle of the launched function (``CUfunction``).
        name: Symbol of the launched function in its CUDA module.
        arguments: The bytes of every argument of the launch, in the order of the function's parameters.
    """

    function: int
    name: str
    arguments: tuple[bytes, ...]


@dataclass(frozen=True)
class MemsetNode:
    """One memory fill of a graph: ``count`` elements of ``width`` bytes at ``pointer`` take ``value``."""

    pointer: int
    value: int
    width: int
    count: int


@dataclass(frozen=True)
class MemcpyNode:
    """One device-to-device copy of a graph: ``size`` bytes from ``source`` to ``destination``."""

    destination: int
    source: int
    size: int


def _driver():
    """The CUDA driver with the prototypes of the graph node queries."""
    global _prototyped
    library = _load_driver()
    if not library:
        raise RuntimeError("The CUDA driver library is unavailable; graph nodes cannot be read")
    if not _prototyped:
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        prototypes = {
            "cuGraphGetNodes": (pointer, ctypes.POINTER(pointer), ctypes.POINTER(size)),
            "cuGraphNodeGetType": (pointer, ctypes.POINTER(ctypes.c_int)),
            "cuGraphNodeGetDependencies": (pointer, ctypes.POINTER(pointer), ctypes.POINTER(size)),
            "cuGraphKernelNodeGetParams_v2": (pointer, ctypes.POINTER(_KernelParams)),
            "cuGraphMemsetNodeGetParams": (pointer, ctypes.POINTER(_MemsetParams)),
            "cuGraphMemcpyNodeGetParams": (pointer, ctypes.POINTER(_MemcpyParams)),
            "cuFuncGetName": (ctypes.POINTER(ctypes.c_char_p), pointer),
            "cuFuncGetParamInfo": (pointer, size, ctypes.POINTER(size), ctypes.POINTER(size)),
        }
        try:
            for name, arguments in prototypes.items():
                function = getattr(library, name)
                function.restype, function.argtypes = ctypes.c_int, arguments
        except AttributeError as error:
            raise RuntimeError(f"The CUDA driver does not provide the graph node queries ({error})") from error
        _prototyped = True
    return library


def _check(code, call):
    if code != 0:
        raise RuntimeError(f"{call} failed (CUDA driver error {code})")


def _handles(query, owner):
    """The node handles that a counted driver query returns for ``owner``."""
    count = ctypes.c_size_t()
    _check(query(owner, None, ctypes.byref(count)), query.__name__)
    if count.value == 0:
        return []
    handles = (ctypes.c_void_p * count.value)()
    _check(query(owner, handles, ctypes.byref(count)), query.__name__)
    return list(handles)


def _kernel_node(library, node):
    params = _KernelParams()
    _check(library.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params)), "cuGraphKernelNodeGetParams_v2")
    if not params.func:
        raise RuntimeError("A kernel node of the graph has no function handle")
    name = ctypes.c_char_p()
    _check(library.cuFuncGetName(ctypes.byref(name), params.func), "cuFuncGetName")
    arguments = []
    offset, size = ctypes.c_size_t(), ctypes.c_size_t()
    # The driver reports an error for the index after the last parameter.
    while library.cuFuncGetParamInfo(params.func, len(arguments), ctypes.byref(offset), ctypes.byref(size)) == 0:
        arguments.append(ctypes.string_at(params.kernelParams[len(arguments)], size.value))
    return KernelNode(params.func, name.value.decode(), tuple(arguments))


def _memset_node(library, node):
    params = _MemsetParams()
    _check(library.cuGraphMemsetNodeGetParams(node, ctypes.byref(params)), "cuGraphMemsetNodeGetParams")
    if params.height != 1:
        raise RuntimeError(f"A memory fill of {params.height} rows is not supported")
    return MemsetNode(params.dst, params.value, params.elementSize, params.width)


def _memcpy_node(library, node):
    params = _MemcpyParams()
    _check(library.cuGraphMemcpyNodeGetParams(node, ctypes.byref(params)), "cuGraphMemcpyNodeGetParams")
    device_memory = (_MEMORY_TYPE_DEVICE, _MEMORY_TYPE_UNIFIED)
    if params.srcMemoryType not in device_memory or params.dstMemoryType not in device_memory:
        raise RuntimeError("A memory copy from or to host memory or a CUDA array is not supported")
    offsets = (params.srcY, params.srcZ, params.srcLOD, params.dstY, params.dstZ, params.dstLOD)
    if params.Height != 1 or params.Depth != 1 or any(offsets):
        raise RuntimeError("A memory copy of more than one row is not supported")
    return MemcpyNode(params.dstDevice + params.dstXInBytes, params.srcDevice + params.srcXInBytes, params.WidthInBytes)


def graph_nodes(graph):
    """Nodes of a captured CUDA graph in the order in which the device executes them.

    Args:
        graph: A ``wp.Graph`` that a CUDA capture returned.

    Returns:
        A list of ``KernelNode``, ``MemsetNode`` and ``MemcpyNode``.

    Raises:
        RuntimeError: If the graph holds a node of another type (memory allocation or
            free, conditional, child graph, ...), if its nodes are not one chain, if a
            memory operation spans more than one row or touches host memory, or if the
            CUDA driver does not provide the queries.
    """
    if not graph.device.is_cuda or not graph.graph:
        raise RuntimeError("Graph nodes can only be read from a captured CUDA graph")
    library = _driver()
    readers = {_KERNEL: _kernel_node, _MEMSET: _memset_node, _MEMCPY: _memcpy_node}
    with graph.device.context_guard:
        handles = _handles(library.cuGraphGetNodes, graph.graph)
        after, first = {}, []
        for handle in handles:
            dependencies = _handles(library.cuGraphNodeGetDependencies, handle)
            if len(dependencies) > 1 or (dependencies and dependencies[0] in after):
                raise RuntimeError("The nodes of the graph are not one chain")
            if dependencies:
                after[dependencies[0]] = handle
            else:
                first.append(handle)
        if len(first) > 1:
            raise RuntimeError("The nodes of the graph are not one chain")
        nodes = []
        handle = first[0] if first else None
        while handle is not None:
            kind = ctypes.c_int()
            _check(library.cuGraphNodeGetType(handle, ctypes.byref(kind)), "cuGraphNodeGetType")
            if kind.value not in readers:
                name = _NODE_TYPES.get(kind.value, f"type {kind.value}")
                raise RuntimeError(f"The graph holds a {name} node; only kernel, memset and memcpy nodes can be read")
            nodes.append(readers[kind.value](library, handle))
            handle = after.get(handle)
        if len(nodes) != len(handles):
            raise RuntimeError("The nodes of the graph are not one chain")
    return nodes
