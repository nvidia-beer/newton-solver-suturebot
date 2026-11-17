# SPDX-License-Identifier: Apache-2.0
"""Reuse launches for numerical contact inputs; retain every host branch/readback.

The first call for a binding runs eagerly. Later calls capture exactly the same
preparation kernels, with current geometry/forces/ALM values read from arrays.
No physical state, solver decision or host dense solve is included in a graph.
"""

import numpy as np
import warp as wp

from ....utils.capture import ScopedCapture, note_release
from ...fem.differentiation import active_tape


def _storage(array):
    return (array.ptr, array.shape, array.strides, array.dtype, array.device.alias)


class _PreparationGraph:
    def __init__(self, device):
        self.device = device
        self.key = self.graph = self.stream = None
        self.owners = ()
        self.captures = self.launches = self.eager = 0

    def invalidate(self):
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError("Retire preparation graphs outside an enclosing CUDA capture")
        # A replacement may arrive on another stream. Retire old array/graph
        # owners only after the stream that launched their work has completed.
        if self.stream is not None:
            wp.synchronize_stream(self.stream)
        if self.graph is not None:
            # A captured graph is dropped here: the census of the process counts it.
            note_release()
        self.key = self.graph = self.stream = None
        self.owners = ()

    def run(self, enqueue, scalars, arrays, *, enabled):
        if self.device.is_cuda and self.device.is_capturing:
            # An enclosing caller owns this eager sequence. Keep our existing
            # cache/array owners alive: retiring them would synchronize during
            # capture, which CUDA forbids, or release in-flight storage.
            enqueue()
            self.eager += 1
            return
        supported = (
            enabled
            and self.device.is_cuda
            and active_tape() is None
            and all(isinstance(a, wp.array) and a.device == self.device and not a.requires_grad for a in arrays)
        )
        if not supported:
            if self.key is not None:
                self.invalidate()
            enqueue()
            self.eager += 1
            return
        stream = wp.get_stream(self.device)
        key = (int(stream.cuda_stream), scalars, tuple(_storage(a) for a in arrays))
        if key != self.key:
            self.invalidate()
            self.key, self.owners, self.stream = key, arrays, stream
            # Real preparation warms modules without evaluating anything twice.
            enqueue()
            self.eager += 1
            return
        if self.graph is None:
            with ScopedCapture(device=self.device, stream=stream) as capture:
                enqueue()
            self.graph = capture.graph
            self.captures += 1
        wp.capture_launch(self.graph, stream=stream)
        self.launches += 1


class ContactPreparation:
    """Capture shaft/combined inputs between native metric and Woodbury binding."""

    def __init__(self, preconditioner):
        self.pre = preconditioner
        self.replay = _PreparationGraph(preconditioner.device)

    def run(self, state, dt, residual):
        from .combined_contact_preconditioner import CombinedContactPreconditioner
        from .shaft_preconditioner import _ShaftPreconditioner

        p, c = self.pre, self.pre.coupler
        interaction, matrix = c.interaction, c.fem_solver.A_bsr
        from ..interface.shaft_projection import projection_binding

        projection = projection_binding(
            interaction, state, c.needle_body, float(interaction.radii.max()), operands=(residual, p.matrices, p.levers)
        )
        alm = interaction.tunnel_alm
        supported_type = type(p) in (_ShaftPreconditioner, CombinedContactPreconditioner)
        original_diagonal = (
            CombinedContactPreconditioner._augment_diagonal
            if type(p) is CombinedContactPreconditioner
            else _ShaftPreconditioner._augment_diagonal
        )
        original_extra = (
            CombinedContactPreconditioner._extra_graph_arrays
            if type(p) is CombinedContactPreconditioner
            else _ShaftPreconditioner._extra_graph_arrays
        )
        supported = (
            getattr(p, "capture_preparation", False)
            and supported_type
            and getattr(p._enqueue_inputs, "__func__", None) is _ShaftPreconditioner._enqueue_inputs
            and getattr(p._augment_diagonal, "__func__", None) is original_diagonal
            and getattr(p._extra_graph_arrays, "__func__", None) is original_extra
            and not getattr(c.model, "requires_grad", False)
            and not c.fem_solver.differentiable
        )
        # The preparation kernels read the time step from p.time_step, whether they are
        # issued, captured or replayed. Outside capture this writes a changed dt; inside
        # an enclosing capture it only checks that the arrays hold dt.
        p.step_time.set(dt)
        if not supported:
            self.replay.run(lambda: p._enqueue_inputs(state, dt, residual), (), (), enabled=False)
            return
        extra = p._extra_graph_arrays()
        arrays = (
            state.particle_q,
            state.particle_qd,
            state.body_q,
            state.body_qd,
            interaction._track_nodes,
            interaction._track_bary,
            interaction._track_lengths,
            interaction._track_engaged,
            # The kernels over the track read the number of its samples from the live counts.
            interaction._live,
            interaction._samples,
            p.matrices,
            p.levers,
            p.shears,
            p.rigid_blocks,
            p.native_damping,
            p.time_step,
            p.fixed,
            matrix.offsets,
            matrix.columns,
            matrix.values,
            p.diagonal,
            p.inverse_diagonal,
            residual,
            p.rhs,
            *extra,
            *((alm.normal, alm.tangent) if alm is not None else ()),
            *(() if projection is None else projection.owners),
        )
        # The time step is not a scalar of the graph: it is the array p.time_step above. Nor is
        # the number of track samples: the launches over the track have the size of its reserve.
        scalars = (
            None if projection is None else projection.key,
            p.capacity,
            p.n,
            p._articulated,
            p.mass,
            tuple(np.asarray(p.inertia).reshape(-1)),
            c.needle_body,
            tuple(c.needle_com),
            c.moment_length,
            float(interaction.radii.max()),
            interaction.friction,
            interaction.speed_scale,
            interaction.shaft_stiffness_per_m,
            interaction.shaft_damping_per_m,
            None if alm is None else (alm.penalty_per_m, alm.compliance_per_m, alm.clearance),
            getattr(getattr(p, "surface", None), "count", None),
        )
        self.replay.run(lambda: p._enqueue_inputs(state, dt, residual), scalars, arrays, enabled=supported)


class PacketPreparation:
    """Capture the two original packet kernels after host basis-binding checks."""

    def __init__(self, guess):
        self.guess = guess
        self.replay = _PreparationGraph(guess.pre.device)

    def run(self, surface_count, surface_valid):
        from .woodbury_guess import WoodburyInitialGuess

        g, p = self.guess, self.guess.pre
        interaction = p.coupler.interaction
        arrays = (
            p.rigid_blocks,
            p.rhs,
            g.core.free,
            g.core.support,
            # The packet kernels read the number of support columns and of track samples from these two.
            g.core.count,
            interaction._live,
            surface_count,
            surface_valid,
            g.packet,
            interaction._track_nodes,
            interaction._track_bary,
            p.matrices,
            p.levers,
            p.shears,
        )
        # No scalar of the graph: both launches have the size of their scratch.
        scalars = ()
        supported = (
            getattr(p, "capture_preparation", False)
            and type(g) is WoodburyInitialGuess
            and getattr(g._enqueue_packet, "__func__", None) is WoodburyInitialGuess._enqueue_packet
            and not getattr(p.coupler.model, "requires_grad", False)
            and not p.coupler.fem_solver.differentiable
        )
        self.replay.run(lambda: g._enqueue_packet(surface_count, surface_valid), scalars, arrays, enabled=supported)
