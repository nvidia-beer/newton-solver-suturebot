# SPDX-License-Identifier: Apache-2.0
"""One frozen shaft plus surface search solve; physical forces remain exact.

Use the sum of both contact tangents inside the existing shaft system. Two
successive inverse corrections are not the inverse of this summed system.
Empty, overflowed, impulse-capped or nonfinite surface recordings have valid=0
and reduce exactly to the shaft-only operator and balanced correction.
"""

import warp as wp

from .shaft_preconditioner import _ShaftPreconditioner
from .surface_preconditioner import surface_balanced_correction, surface_contact_action, surface_contact_diagonal


class CombinedContactPreconditioner(_ShaftPreconditioner):
    """Add recorded facet curvature to the unchanged native shaft metric."""

    def __init__(self, coupler, surface):
        self.surface = surface
        super().__init__(coupler)

    def _surface_arguments(self):
        # Use this metric's dt: the standalone surface.apply is not invoked
        # while the shaft and surface share a single Krylov solve.
        a = self.surface.anchors
        return [self.fixed, a.count, self.surface.valid, a.nodes, a.bary, a.matrices, a.levers, self.time_step]

    def _augment_contact_action(self, x):
        wp.launch(
            surface_contact_action,
            self.surface.count,
            self._surface_arguments() + [x, self.work],
            device=self.device,
        )

    def _augment_diagonal(self):
        wp.launch(
            surface_contact_diagonal,
            self.surface.count,
            self._surface_arguments() + [self.diagonal],
            device=self.device,
        )

    def _augment_correction(self, velocity, output):
        wp.launch(
            surface_balanced_correction,
            self.surface.count,
            self._surface_arguments() + [velocity, output],
            device=self.device,
        )

    def _extra_graph_arrays(self):
        # Pointer/shape/dtype in the base key and retained strong owners cover
        # fixed-capacity recorder replacement without a new contact count.
        return tuple(self._surface_arguments())
