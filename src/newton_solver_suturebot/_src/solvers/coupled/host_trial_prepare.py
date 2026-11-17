# SPDX-License-Identifier: Apache-2.0
"""Capture unchanged load staging and full-state copies before native trials."""

from collections import OrderedDict

import newton
import warp as wp

from ...utils.capture import ScopedCapture, note_release
from ..fem.differentiation import active_tape
from .coupling import _apply_interface_load


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

    The first use of the helper runs the preparation as written, and the next use
    of a binding captures it. :meth:`capture_graph` does both for one start state
    ahead of the trials, so that the trials of that state only replay.
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
                with ScopedCapture(device=self.device) as capture:
                    operation()
                entry = (capture.graph, arrays)
                self.graphs[key] = entry
                if len(self.graphs) > 8:
                    self.graphs.popitem(last=False)
                    note_release()
                self.captures += 1
            wp.capture_launch(entry[0])
            self.launches += 1

        # The binding this callable replays; a callable without one runs the preparation as written.
        replay.key = key
        return replay

    def capture_graph(self, state_in):
        """Leave a captured and launched graph for the binding of ``state_in``; for start-up.

        Runs the preparation of ``state_in`` once as written if this helper never ran, and
        once more into a capture if the binding has no graph; the new graph is launched,
        which is where Warp builds its executable. A binding with a graph is left alone.
        Every pass is the real preparation: it writes the trial forces, the incoming body
        forces and the candidate state, as a trial does, and the caller restores what its
        attempt owns. ``captures`` counts the capture; ``launches`` keeps its value.

        Returns:
            False if the configuration runs the preparation as written (see :meth:`bind`);
            nothing is issued then.
        """
        replay = self.bind(state_in)
        key = getattr(replay, "key", None)
        if key is None:
            return False
        launches = self.launches
        try:
            if not self.warmed:
                replay()
            if key not in self.graphs:
                replay()
        finally:
            self.launches = launches
        return True
