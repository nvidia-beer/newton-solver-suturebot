# SPDX-License-Identifier: Apache-2.0
"""Exact forward-only tables for the authored needle polyline.

The one-thread refresh preserves the original sequential float32 arc sums.
It detects in-place edits without a host readback. Graph owners must retain the
table and source array; replacing source storage requires a new graph binding.
No state-dependent projection, force or multiplier is cached.
"""

import warp as wp


@wp.func_native("""
    #if defined(__CUDA_ARCH__)
        return __float_as_int(value);
    #else
        union { float f; int i; } bits;
        bits.f = value;
        return bits.i;
    #endif
""")
def _float_bits(value: float) -> int: ...


@wp.struct
class ProjectionTable:
    source: wp.array[wp.vec3]
    samples: wp.array[wp.vec3]
    directions: wp.array[wp.vec3]
    lengths: wp.array[float]
    denominators: wp.array[float]
    arcs: wp.array[float]
    initialized: wp.array[int]
    valid: wp.array[int]
    generation: wp.array[int]


@wp.kernel(enable_backward=False)
def refresh_table(source: wp.array[wp.vec3], table: ProjectionTable):
    changed = table.initialized[0] == 0
    if table.initialized[0] != 0:
        for j in range(source.shape[0]):
            a, b = source[j], table.samples[j]
            # Bitwise tests also detect signed-zero changes. NaNs are compared
            # by representation; nonfinite segments use the original path.
            for axis in range(3):
                if _float_bits(a[axis]) != _float_bits(b[axis]):
                    changed = True
    if changed:
        valid = source.shape[0] >= 2
        arc = float(0.0)
        table.arcs[0] = arc
        for j in range(source.shape[0]):
            table.samples[j] = source[j]
            if j < source.shape[0] - 1:
                direction = source[j + 1] - source[j]
                length = wp.length(direction)
                denominator = wp.dot(direction, direction)
                table.directions[j] = direction
                table.lengths[j] = length
                table.denominators[j] = denominator
                arc += length
                table.arcs[j + 1] = arc
                if not (length > 0.0 and wp.isfinite(length) and wp.isfinite(arc)):
                    valid = False
        table.generation[0] += 1
        table.valid[0] = int(valid)
        table.initialized[0] = 1


@wp.func
def _arc_point(value: float, source: wp.array[wp.vec3], table: ProjectionTable):
    point = wp.vec3(0.0)
    if value >= 0.0 and value <= table.arcs[source.shape[0] - 1]:
        lower = int(0)
        upper = int(source.shape[0] - 1)
        # Rightmost start <= value preserves the original inclusive intervals
        # and last-match-wins rule, including repeated rounded arc prefixes.
        while lower < upper:
            middle = (lower + upper) // 2
            if table.arcs[middle] <= value:
                lower = middle + 1
            else:
                upper = middle
        segment = wp.max(0, lower - 1)
        start = table.arcs[segment]
        start_point, end_point = source[segment], source[segment + 1]
        segment_length = wp.length(end_point - start_point)
        if start <= value and value <= start + segment_length:
            point = start_point + (value - start) / segment_length * (end_point - start_point)
    return point


class ShaftProjectionTables:
    """Owns one source binding and immutable-size scratch; eager rebind is safe.

    Call refresh before every captured/eager projection batch. Keep the returned
    table with its captured graph. Rebinding never mutates old table storage.
    This forward-only prototype is not an autodifferentiation entry point.
    """

    def __init__(self, samples):
        self.source = None
        self.table = None
        self.bind(samples)

    def bind(self, samples):
        if samples.dtype != wp.vec3 or samples.ndim != 1 or not samples.is_contiguous:
            raise ValueError("Expected contiguous vec3 needle samples")
        if samples.requires_grad:
            raise ValueError("Use the original projection for differentiable samples")
        if samples.device.is_capturing:
            raise RuntimeError("Bind needle projection storage before graph capture")
        if samples is not self.source:
            table = ProjectionTable()
            table.source = samples
            count = len(samples)
            table.samples = wp.empty_like(samples)
            table.directions = wp.empty(max(0, count - 1), dtype=wp.vec3, device=samples.device)
            table.lengths = wp.empty(max(0, count - 1), dtype=float, device=samples.device)
            table.denominators = wp.empty_like(table.lengths)
            table.arcs = wp.empty(max(1, count), dtype=float, device=samples.device)
            table.initialized = wp.zeros(1, dtype=int, device=samples.device)
            table.valid = wp.zeros(1, dtype=int, device=samples.device)
            table.generation = wp.zeros(1, dtype=int, device=samples.device)
            self.source, self.table = samples, table
        return self.table

    def refresh(self):
        wp.launch(refresh_table, 1, [self.source, self.table], device=self.source.device)
        return self.table
