# SPDX-License-Identifier: Apache-2.0
"""Capture unchanged load staging and full-state copies before native trials."""

from collections import OrderedDict

import newton
import warp as wp

from .coupling import _apply_interface_load
from .solvers.fem.differentiation import active_tape


def _arrays(state):
    """Match the top-level and namespaced arrays copied by Newton State.assign."""
    values = []
    for name, value in vars(state).items():
        if isinstance(value, wp.array):
            values.append((name, value))
        elif isinstance(value, newton.Model.AttributeNamespace):
            values.extend(
                (f"{name}.{field}", array) for field, array in vars(value).items() if isinstance(array, wp.array)
            )
    return tuple(sorted(values))


def _storage(array):
    return (array.ptr, array.shape, array.strides, array.dtype)


class CapturedTrialPreparation:
    """Bind once per host solve; replay once per actual mechanics trial.

    No state field is omitted, including custom namespaces and history arrays.
    Arrays and scalar load mappings must stay fixed until the bound callable
    finishes; their values may change on the same stream. A later bind checks
    storage again. The caller still owns native rollback and physical history.
    """

    def __init__(self, coupler):
        self.c = coupler
        self.device = coupler.model.device
        self.graphs = OrderedDict()
        self.warmed = False
        self.captures = self.launches = 0

    def bind(self, state_in):
        c, candidate = self.c, self.c._candidate

        def operation():
            wp.launch(
                _apply_interface_load,
                max(c.model.particle_count, c.model.body_count),
                [
                    c._trial_load,
                    c._external_nodes,
                    c._external_wrench,
                    c.needle_body,
                    c.moment_length,
                    c._forces,
                    state_in.body_f,
                ],
                device=self.device,
            )
            candidate.assign(state_in)

        if (
            not c.capture_fem
            or not self.device.is_cuda
            or self.device.is_capturing
            or active_tape() is not None
            or getattr(c.model, "requires_grad", False)
            or type(state_in) is not newton.State
            or type(candidate) is not newton.State
            or getattr(candidate.assign, "__func__", None) is not newton.State.assign
        ):
            return operation
        incoming, outgoing = _arrays(state_in), _arrays(candidate)
        if tuple(name for name, _ in incoming) != tuple(name for name, _ in outgoing):
            # Preserve State.assign's own mismatch errors on the ordinary path.
            return operation
        buffers = (c._trial_load, c._external_nodes, c._external_wrench, c._forces)
        arrays = (*buffers, *(array for _, array in incoming), *(array for _, array in outgoing))
        if any(array.requires_grad or array.device != self.device for array in arrays):
            return operation
        key = (
            c.model.particle_count,
            c.model.body_count,
            c.needle_body,
            c.moment_length,
            tuple(_storage(array) for array in buffers),
            tuple((name, _storage(array)) for name, array in incoming),
            tuple((name, _storage(array)) for name, array in outgoing),
        )
        entry = self.graphs.get(key)
        if entry is not None:
            self.graphs.move_to_end(key)

        def replay():
            nonlocal entry
            if not self.warmed:
                operation()
                self.warmed = True
                return
            if entry is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    operation()
                entry = (capture.graph, arrays)
                self.graphs[key] = entry
                if len(self.graphs) > 8:
                    self.graphs.popitem(last=False)
                self.captures += 1
            wp.capture_launch(entry[0])
            self.launches += 1

        return replay
