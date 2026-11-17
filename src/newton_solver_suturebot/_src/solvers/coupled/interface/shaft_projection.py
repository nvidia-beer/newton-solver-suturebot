# SPDX-License-Identifier: Apache-2.0
"""Forward-only CUDA query batch: one warp cooperates on one authored polyline.

This is an isolated proposal. No physical state, force or multiplier is cached.
The arc table of the needle samples is built once, when a binding is made, and
retained with graphs; no batch refreshes it. The samples are the authored geometry
of one interaction: other storage for them is another binding, and an edit in
place is taken up by ProjectionBinding.refresh_table.
"""

import warp as wp

from ...fem.differentiation import active_tape
from .insertion import LIVE_TRACK, _shaft_projection
from .shaft_projection_tables import ProjectionTable, ShaftProjectionTables, _arc_point, _float_bits


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_down_sync(0xffffffff, value, offset);
#else
    return value;
#endif
""")
def shuffle_float(value: float, offset: int) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_down_sync(0xffffffff, value, offset);
#else
    return value;
#endif
""")
def shuffle_int(value: int, offset: int) -> int: ...


@wp.func
def arc_segment(value: float, samples: wp.array[wp.vec3], table: ProjectionTable):
    selected = int(-1)
    if value >= 0.0 and value <= table.arcs[samples.shape[0] - 1]:
        lower, upper = int(0), int(samples.shape[0] - 1)
        while lower < upper:
            middle = (lower + upper) // 2
            if table.arcs[middle] <= value:
                lower = middle + 1
            else:
                upper = middle
        segment = wp.max(0, lower - 1)
        length = wp.length(samples[segment + 1] - samples[segment])
        if table.arcs[segment] <= value and value <= table.arcs[segment] + length:
            selected = segment
    return selected


@wp.kernel(enable_backward=False)
def cooperative_projection(
    queries: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    radii: wp.array[float],
    samples: wp.array[wp.vec3],
    table: ProjectionTable,
    result: wp.array2d[float],
    branches: wp.array2d[int],
    dirty: wp.array[int],
    initialized: wp.array[int],
    generations: wp.array[int],
    counts: wp.array[int],
):
    tid = wp.tid()
    query, lane = tid // 32, tid % 32
    if query < queries.shape[0] and dirty[query] != 0:
        point, pose, radius = queries[query], poses[query], radii[query]
        usable = table.valid[0] != 0 and wp.isfinite(radius) and radius >= 0.0
        if usable:
            local = wp.transform_point(wp.transform_inverse(pose), point)
            best = float(1.0e20)
            index = int(2147483647)
            for segment in range(lane, samples.shape[0] - 1, 32):
                a, b = samples[segment], samples[segment + 1]
                direction = b - a
                fraction = wp.dot(local - a, direction) / wp.max(wp.dot(direction, direction), 1.0e-20)
                candidate = a + wp.clamp(fraction, 0.0, 1.0) * direction
                distance = wp.length_sq(candidate - local)
                if distance < best or (distance == best and segment < index):
                    best, index = distance, segment
            for shift in range(5):
                offset = 16 >> shift
                other_distance = shuffle_float(best, offset)
                other_index = shuffle_int(index, offset)
                if lane + offset < 32 and (other_distance < best or (other_distance == best and other_index < index)):
                    best, index = other_distance, other_index
            if lane == 0:
                if best < 1.0e20 and index < samples.shape[0] - 1:
                    start_point, end_point = samples[index], samples[index + 1]
                    direction = end_point - start_point
                    fraction = wp.dot(local - start_point, direction) / wp.max(wp.dot(direction, direction), 1.0e-20)
                    closest = start_point + wp.clamp(fraction, 0.0, 1.0) * direction
                    length = wp.length(direction)
                    arc = table.arcs[index] + wp.clamp(fraction, 0.0, 1.0) * length
                    before = wp.max(0.0, arc - radius)
                    after = wp.min(table.arcs[samples.shape[0] - 1], arc + radius)
                    lower, upper = (
                        _arc_point(before, samples, table),
                        _arc_point(after, samples, table),
                    )
                    tangent = wp.normalize(direction)
                    use_secant = wp.length_sq(upper - lower) > 1.0e-20
                    if use_secant:
                        tangent = wp.normalize(upper - lower)
                    covered = not (
                        (index == 0 and fraction < 0.0) or (index == samples.shape[0] - 2 and fraction > 1.0)
                    )
                    world = wp.transform_point(pose, closest)
                    tangent = wp.quat_rotate(wp.transform_get_rotation(pose), tangent)
                    for axis in range(3):
                        result[query, axis] = world[axis]
                        result[query, axis + 3] = tangent[axis]
                    result[query, 6] = best
                    result[query, 7] = float(covered)
                    branches[query, 0] = index
                    branches[query, 1] = int(fraction < 0.0) - int(fraction > 1.0)
                    branches[query, 2] = arc_segment(before, samples, table)
                    branches[query, 3] = arc_segment(after, samples, table)
                    branches[query, 4] = int(use_secant)
                    branches[query, 5] = int(covered)
                else:
                    usable = False
        if lane == 0 and not usable:
            closest, tangent, distance, covered = _shaft_projection(point, pose, samples, radius)
            for axis in range(3):
                result[query, axis] = closest[axis]
                result[query, axis + 3] = tangent[axis]
            result[query, 6] = distance
            result[query, 7] = float(covered)
            for j in range(6):
                branches[query, j] = -1

    if query < queries.shape[0] and lane == 0 and dirty[query] != 0:
        initialized[query] = 1
        generations[query] = table.generation[0]
        wp.atomic_add(counts, 0, 1)


@wp.kernel(enable_backward=False)
def prepare_queries(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    live: wp.array[int],
    q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body: int,
    radius: float,
    table: ProjectionTable,
    queries: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    radii: wp.array[float],
    initialized: wp.array[int],
    generations: wp.array[int],
    dirty: wp.array[int],
    counts: wp.array[int],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        dirty[i] = 0
        return
    point = wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
    pose = body_q[body]
    changed = initialized[i] == 0 or generations[i] != table.generation[0]
    if initialized[i] != 0:
        for axis in range(3):
            if _float_bits(point[axis]) != _float_bits(queries[i][axis]):
                changed = True
        for axis in range(7):
            if _float_bits(pose[axis]) != _float_bits(poses[i][axis]):
                changed = True
        if _float_bits(radius) != _float_bits(radii[i]):
            changed = True
    dirty[i] = int(changed)
    queries[i] = point
    poses[i] = pose
    radii[i] = radius
    if not changed:
        wp.atomic_add(counts, 1, 1)


@wp.func
def read_projection(result: wp.array2d[float], i: int):
    return (
        wp.vec3(result[i, 0], result[i, 1], result[i, 2]),
        wp.vec3(result[i, 3], result[i, 4], result[i, 5]),
        result[i, 6],
        result[i, 7] != 0.0,
    )


class ProjectionBinding:
    """One immutable storage binding, with exact device freshness checks.

    The numerical result belongs only to its current point/pose/sample bits.
    Every consumer enqueues freshness checks, including captured graph replay.
    Forces, engagement and ALM multipliers are never retained in this packet.

    The packet has one row per track sample of the interaction's track reserve, and both
    kernels are launched over the reserve: a new track sample changes neither the binding
    nor its key nor a launch size. A row from the live track count on is never marked
    dirty, so nothing is projected or counted for it and its result is not written.

    The arc table of the needle samples is built here, once, outside every capture: a
    batch (``ensure``) issues the two query kernels and nothing for the table. The needle
    samples are constants of a binding. Other storage for them makes another binding (the
    key holds their storage); after an edit in place, ``refresh_table`` rebuilds the table
    and marks every row for a new projection.
    """

    def __init__(self, interaction, state, body, radius):
        self.device = interaction.device
        self.nodes, self.bary = interaction._track_nodes, interaction._track_bary
        # The interaction's device array of live counts: the bound of prepare_queries.
        self.live = interaction._live
        self.q, self.body_q = state.particle_q, state.body_q
        self.samples = interaction._samples
        self.body, self.radius = body, radius
        # Rows of the packet and launch size: the track reserve, not the live track count.
        self.capacity = interaction.track_capacity
        self.stream = int(wp.get_stream(self.device).cuda_stream) if self.device.is_cuda else None
        self.tables = ShaftProjectionTables(self.samples)
        # Built now, with the binding: the kernel module is loaded here and no batch refreshes the table.
        self.tables.refresh()
        n = self.capacity
        self.queries = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.poses = wp.zeros(n, dtype=wp.transform, device=self.device)
        self.radii = wp.zeros(n, dtype=float, device=self.device)
        self.result = wp.zeros((n, 8), dtype=float, device=self.device)
        self.branches = wp.zeros((n, 6), dtype=int, device=self.device)
        self.initialized = wp.zeros(n, dtype=int, device=self.device)
        self.generations = wp.zeros(n, dtype=int, device=self.device)
        self.dirty = wp.zeros(n, dtype=int, device=self.device)
        self.counts = getattr(interaction, "_cooperative_shaft_counts", None)
        if self.counts is None:
            self.counts = wp.zeros(2, dtype=int, device=self.device)
            interaction._cooperative_shaft_counts = self.counts
        self.key = self.binding_key(interaction, state, body, radius)

    @staticmethod
    def binding_key(interaction, state, body, radius):
        """Storage and settings of a binding. The live track count is device data and no part of it."""
        arrays = (
            interaction._track_nodes,
            interaction._track_bary,
            interaction._live,
            interaction._samples,
            state.particle_q,
            state.body_q,
        )
        return (
            body,
            float(radius),
            tuple((id(a), a.ptr, a.shape, a.strides, a.dtype) for a in arrays),
        )

    def stale(self, interaction):
        """Whether the interaction replaced the track storage that this binding addresses."""
        return (
            self.nodes is not interaction._track_nodes
            or self.bary is not interaction._track_bary
            or self.live is not interaction._live
            or self.samples is not interaction._samples
        )

    def serves(self, interaction, state, body, radius):
        """Whether these are the settings and the very arrays of this binding's key.

        The six arrays of the key are compared by identity. The binding holds them, so an equal
        identity is the same storage, shape and type, and ``binding_key`` of these arguments is
        this binding's key.
        """
        return (
            self.q is state.particle_q
            and self.body_q is state.body_q
            and self.key[0] == body
            and self.key[1] == float(radius)
            and not self.stale(interaction)
        )

    @property
    def owners(self):
        t = self.tables.table
        return (
            self.nodes,
            self.bary,
            self.live,
            self.q,
            self.body_q,
            self.samples,
            t.samples,
            t.directions,
            t.lengths,
            t.denominators,
            t.arcs,
            t.initialized,
            t.valid,
            t.generation,
            self.queries,
            self.poses,
            self.radii,
            self.result,
            self.branches,
            self.initialized,
            self.generations,
            self.dirty,
            self.counts,
        )

    def refresh_table(self):
        """Rebuild the arc table from the needle samples, after they were edited in place.

        One launch of the one-thread refresh kernel, which compares the samples with the copy
        in the table. A changed sample advances the table's generation, so the next batch
        projects every live row again, also in a graph that was captured before.
        """
        return self.tables.refresh()

    def ensure(self):
        # The table of the binding's needle samples, built with the binding (see the class).
        table = self.tables.table
        wp.launch(
            prepare_queries,
            self.capacity,
            [
                self.nodes,
                self.bary,
                self.live,
                self.q,
                self.body_q,
                self.body,
                self.radius,
                table,
                self.queries,
                self.poses,
                self.radii,
                self.initialized,
                self.generations,
                self.dirty,
                self.counts,
            ],
            device=self.device,
        )
        n = ((self.capacity * 32 + 255) // 256) * 256
        wp.launch(
            cooperative_projection,
            n,
            [
                self.queries,
                self.poses,
                self.radii,
                self.samples,
                table,
                self.result,
                self.branches,
                self.dirty,
                self.initialized,
                self.generations,
                self.counts,
            ],
            device=self.device,
            block_dim=256,
        )


# Bindings that one interaction keeps: the candidate state and the two alternating accepted-state buffers.
MAXIMUM_BINDINGS = 3


def projection_binding(interaction, state, body, radius, *, operands=()):
    """Return an opted-in CUDA binding, or retain the original projection path.

    Capturing callers prebind outside capture and include owners in graph keys.
    No binding is made within an enclosing capture: a capture that asks for a key
    that was not bound raises, because its graph would keep the original kernel
    for as long as it is replayed (the track count is no part of a graph key).
    CPU, tapes, any gradient state and another stream than the bindings' stay original.

    A binding exists from an empty track on and does not follow the track count. The
    interaction keeps one per key in ``_cooperative_shaft_bindings``, at most
    ``MAXIMUM_BINDINGS``. A new key first drops the entries whose track storage the
    interaction replaced, then the least recently returned one, outside a capture
    only; each drop is counted in ``interaction._cooperative_shaft_rebinds``.
    ``interaction._cooperative_shaft_binding`` is the binding returned last.

    Every call checks the opt-in, the tape and every array of the state and of the
    operands, in one pass. A call for the very arrays and settings of the binding
    returned last gets that binding without forming a key: the call of every
    substep of a consumer that prebinds one state.
    """
    device = interaction.device
    if (
        not getattr(interaction, "cooperative_shaft_projection", False)
        or not device.is_cuda
        or active_tape() is not None
    ):
        return None
    samples, nodes, bary = interaction._samples, interaction._track_nodes, interaction._track_bary
    for value in vars(state).values():
        if isinstance(value, wp.array) and (value.requires_grad or value.device != device):
            return None
    for value in (samples, nodes, bary, *operands):
        if value.requires_grad or value.device != device:
            return None
    capacity = getattr(interaction, "track_capacity", None)
    if (
        nodes.size != capacity
        or bary.size != capacity
        or samples.dtype != wp.vec3
        or samples.ndim != 1
        or not samples.is_contiguous
    ):
        return None
    bindings = getattr(interaction, "_cooperative_shaft_bindings", None)
    if bindings is None:
        bindings = interaction._cooperative_shaft_bindings = {}
        interaction._cooperative_shaft_rebinds = 0
    capturing = device.is_capturing
    stream = int(wp.get_stream(device).cuda_stream)
    other_stream = any(binding.stream != stream for binding in bindings.values())
    if other_stream and not capturing:
        # Keep the old owners alive and retain the original projection on a
        # different stream. This packet has one ordered-stream contract.
        return None
    binding = getattr(interaction, "_cooperative_shaft_binding", None)
    if (
        binding is not None
        and binding.serves(interaction, state, body, radius)
        and next(reversed(bindings.values()), None) is binding
    ):
        # The binding returned last, asked for again: it is the last entry already.
        return binding
    key = ProjectionBinding.binding_key(interaction, state, body, radius)
    binding = bindings.pop(key, None)
    if binding is None:
        if capturing:
            if other_stream:
                # As outside a capture: the bindings belong to another stream.
                return None
            raise RuntimeError(
                "The cooperative shaft projection is enabled, but no projection is bound for the state of "
                "this capture: call projection_binding(interaction, state, body, radius) outside the capture"
            )
        dropped = [other for other, kept in bindings.items() if kept.stale(interaction)]
        current = [other for other in bindings if other not in dropped]
        # Room for the new binding: the least recently returned entries go first.
        dropped += current[: max(0, len(current) - MAXIMUM_BINDINGS + 1)]
        for other in dropped:
            del bindings[other]
        interaction._cooperative_shaft_rebinds += len(dropped)
        binding = ProjectionBinding(interaction, state, body, radius)
    # Last in the dictionary: the most recently returned binding.
    bindings[key] = binding
    interaction._cooperative_shaft_binding = binding
    return binding


def prepare_projection(interaction, state, body, radius, *, operands=()):
    binding = projection_binding(interaction, state, body, radius, operands=operands)
    if binding is not None:
        binding.ensure()
    return binding
