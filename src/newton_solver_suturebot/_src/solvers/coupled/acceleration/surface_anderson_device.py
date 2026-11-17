# SPDX-License-Identifier: Apache-2.0
"""Device-resident Anderson history with exact sparse transport to the CPU SVD.

No normal equations, thresholds, force laws or physical acceptance tests change.
Only exactly-zero rows are omitted. CPU reconstruction keeps the original full
residual norm; active rows reach the existing thin SVD in their original order.
"""

import numpy as np
import warp as wp
import warp._src.context as warp_context

from .surface_anderson import SurfaceAnderson

_SLOTS = 8
_SlotVector = wp.types.vector(length=_SLOTS, dtype=int)
_FIELDS = 3 * _SLOTS + 1


@wp.kernel(enable_backward=False)
def _pack(
    load: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    direction: wp.array[wp.float64],
    history: wp.array3d[wp.float64],
    slots: _SlotVector,
    count: int,
    current_slot: int,
    pending: wp.array[wp.float64],
    has_pending: int,
    capacity: int,
    packet: wp.array[wp.float64],
    total: wp.array[int],
    row_slot: wp.array[int],
):
    row = wp.tid()
    a, b, c = load[row], residual[row], direction[row]
    history[current_slot, 0, row] = a
    history[current_slot, 1, row] = b
    history[current_slot, 2, row] = c
    active = a != wp.float64(0.0) or b != wp.float64(0.0) or c != wp.float64(0.0)
    for past in range(count):
        for field in range(3):
            active = active or history[slots[past], field, row] != wp.float64(0.0)
    if has_pending != 0:
        active = active or pending[row] != wp.float64(0.0)
    row_slot[row] = -1
    if active:
        destination = wp.atomic_add(total, 0, 1)
        row_slot[row] = destination
        if destination < capacity:
            packet[1 + destination] = wp.float64(row)
            for field in range(3):
                packet[1 + (1 + field) * capacity + destination] = history[current_slot, field, row]
            for past in range(count):
                for field in range(3):
                    packet[1 + (4 + 3 * past + field) * capacity + destination] = history[slots[past], field, row]
            value = wp.float64(0.0)
            if has_pending != 0:
                value = pending[row]
            packet[1 + (1 + 3 * _SLOTS) * capacity + destination] = value


@wp.kernel(enable_backward=False)
def _header(total: wp.array[int], packet: wp.array[wp.float64]):
    packet[0] = wp.float64(total[0])


@wp.kernel(enable_backward=False)
def _scatter_response(
    mapping: wp.array[int],
    capacity: int,
    response: wp.array[wp.float64],
    apply: int,
    has_pending: int,
    load: wp.array[wp.float64],
    pending: wp.array[wp.float64],
):
    row = wp.tid()
    slot = mapping[row]
    proposal, fallback = wp.float64(0.0), wp.float64(0.0)
    if slot >= 0 and slot < capacity:
        proposal = response[slot]
        fallback = response[capacity + slot]
    if apply != 0:
        load[row] = proposal
    if has_pending != 0:
        pending[row] = fallback


class DeviceSurfaceAnderson(SurfaceAnderson):
    """Keep authoritative history on the device; use the original CPU thin SVD.

    Supported callers run this eagerly between physical trials. Tape/gradient
    use fails closed so the caller can retain its original CPU implementation.
    A bounded packet overflow grows the retained capacity before any history
    or proposal is committed. Future calls reuse that larger packet. All nonzero values, including subnormals, survive.
    The packet of the configured capacity is allocated with the workspace, so the first
    proposal allocates nothing unless its rows overflow that capacity.
    """

    def __init__(self, template, *, capacity=128):
        if template.dtype != wp.float64 or template.ndim != 1 or not template.is_contiguous:
            raise ValueError("Device Anderson requires a contiguous float64 interface vector")
        if capacity < 1:
            raise ValueError("Packet capacity must be positive")
        self.device, self.size = template.device, template.size
        self._storage = wp.zeros((_SLOTS, 3, self.size), dtype=wp.float64, device=self.device)
        self._fallback = wp.zeros(self.size, dtype=wp.float64, device=self.device)
        self._row_slot = wp.empty(self.size, dtype=int, device=self.device)
        self._total = wp.zeros(1, dtype=int, device=self.device)
        self._packets = {}
        self._capacity = min(capacity, max(1, self.size))
        self._norm_workspace = np.zeros(self.size, dtype=np.float64)
        self.transport_calls = self.transport_bytes = self.overflows = self.max_active_rows = 0
        self._buffers(self._capacity)
        super().__init__(depth=_SLOTS - 2)

    def reset(self):
        super().reset()
        self._slots = []
        self._norms = []
        self._pending_metadata = None
        self._norm_map = {}
        self._current_norm = 0.0
        self._host_load_witness = None
        self._host_wrench_scope = None

    def _residual_norm(self, residual):
        # Existing history views have cached full-vector norms. The only new
        # copy SurfaceAnderson creates is the current residual appended below.
        return self._norm_map.get(id(residual), self._current_norm)

    def _buffers(self, capacity):
        if capacity not in self._packets:
            self._packets[capacity] = (
                wp.empty(1 + (1 + _FIELDS) * capacity, dtype=wp.float64, device=self.device),
                wp.empty(2 * capacity, dtype=wp.float64, device=self.device),
                np.zeros(2 * capacity, dtype=np.float64),
            )
        return self._packets[capacity]

    def _read(self, load, residual, direction, current_slot, capacity):
        packet, response, host_response = self._buffers(capacity)
        slots = _SlotVector(*(self._slots + [0] * (_SLOTS - len(self._slots))))
        self._total.zero_()
        wp.launch(
            _pack,
            self.size,
            [
                load,
                residual,
                direction,
                self._storage,
                slots,
                len(self._slots),
                current_slot,
                self._fallback,
                int(self._pending_metadata is not None),
                capacity,
                packet,
                self._total,
                self._row_slot,
            ],
            device=self.device,
        )
        wp.launch(_header, 1, [self._total, packet], device=self.device)
        values = packet.numpy()
        self.transport_calls += 1
        self.transport_bytes += values.nbytes
        count = int(values[0])
        self.max_active_rows = max(self.max_active_rows, count)
        if count > capacity:
            return None
        indices = values[1 : 1 + capacity][:count].astype(np.int64)
        order = np.argsort(indices, kind="stable")
        fields = values[1 + capacity :].reshape(_FIELDS, capacity)[:, :count][:, order].copy()
        return indices[order], order, fields, response, host_response

    def propose_device(self, load, residual, relaxation, direction=None):
        """Return True only when a proposal/recovery load was written in place."""
        host_scope, self._host_wrench_scope = self._host_wrench_scope, None
        self._host_load_witness = None
        explicit_direction = direction is not None
        if direction is None:
            direction = residual
        if warp_context.runtime.tape is not None:
            raise RuntimeError("Device Anderson transport is not differentiable")
        for array in (load, residual, direction):
            if (
                array.dtype != wp.float64
                or array.shape != (self.size,)
                or array.device != self.device
                or array.requires_grad
                or not array.is_contiguous
            ):
                raise ValueError("Unsupported Anderson interface storage")
        current_slot = next(slot for slot in range(_SLOTS) if slot not in self._slots)
        capacity = self._capacity
        packed = self._read(load, residual, direction, current_slot, capacity)
        if packed is None:
            self.overflows += 1
            # The header reports the complete exact union even when payload
            # storage overflows. Grow once to cover it, preserving all rows;
            # retain the capacity across calls and history resets.
            capacity = min(self.size, 1 << (self.max_active_rows - 1).bit_length())
            self._capacity = capacity
            packed = self._read(load, residual, direction, current_slot, capacity)
            if packed is None:
                raise RuntimeError("Grown Anderson transport unexpectedly overflowed")
        indices, order, fields, response, host_response = packed
        current = tuple(fields[field] for field in range(3))
        self._norm_workspace.fill(0.0)
        self._norm_workspace[indices] = current[1]
        self._current_norm = np.linalg.norm(self._norm_workspace)
        self.history = [tuple(fields[3 + 3 * past + field] for field in range(3)) for past in range(len(self._slots))]
        originals = {
            id(item): (slot, norm) for item, slot, norm in zip(self.history, self._slots, self._norms, strict=True)
        }
        self._norm_map = {id(item[1]): norm for item, norm in zip(self.history, self._norms, strict=True)}
        if self._pending_metadata is not None:
            self.pending = (*self._pending_metadata, fields[-1])
        if explicit_direction:
            proposal = super().propose(*current[:2], relaxation, direction=current[2])
        else:
            proposal = super().propose(*current[:2], relaxation)
        new_slots, new_norms = [], []
        for item in self.history:
            slot, norm = originals.get(id(item), (current_slot, self._current_norm))
            new_slots.append(slot)
            new_norms.append(norm)
        pending = self.pending
        self._slots, self._norms = new_slots, new_norms
        self._pending_metadata = None if pending is None else pending[:3]
        # Never retain full/compact NumPy history as authoritative state.
        self.history, self.pending, self._norm_map = [], None, {}
        if proposal is None and pending is None:
            return False
        host_response.fill(0.0)
        if proposal is not None:
            host_response[order] = proposal
        if pending is not None:
            host_response[capacity + order] = pending[3]
        response.assign(host_response)
        wp.launch(
            _scatter_response,
            self.size,
            [
                self._row_slot,
                capacity,
                response,
                int(proposal is not None),
                int(pending is not None),
                load,
                self._fallback,
            ],
            device=self.device,
        )
        if proposal is not None and self.size >= 6 and self.device.is_cuda and host_scope is not None:
            from ..host_wrench import HostLoadWitness

            self._host_load_witness = HostLoadWitness.from_sparse(load, indices, proposal, host_scope)
        return proposal is not None

    def bind_host_wrench_scope(self, scope):
        # Existing subclasses may replace the proposal and its upload ordering.
        # Keep their original signature and native-import path unchanged.
        supported = getattr(self.propose_device, "__func__", None) is _HOST_WRENCH_PROPOSAL
        self._host_wrench_scope = scope if supported else None
        self._host_load_witness = None
        return supported

    def take_host_load_witness(self):
        witness, self._host_load_witness = self._host_load_witness, None
        return witness

    def diagnostics(self):
        return dict(
            transport_calls=self.transport_calls,
            transport_bytes=self.transport_bytes,
            overflow_packets=self.overflows,
            max_active_rows=self.max_active_rows,
            packet_capacity=self._capacity,
            committed_history=len(self._slots),
            proposals=self.proposals,
            rejections=self.rejections,
        )


_HOST_WRENCH_PROPOSAL = DeviceSurfaceAnderson.propose_device
