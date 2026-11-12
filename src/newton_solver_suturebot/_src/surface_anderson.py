# SPDX-License-Identifier: Apache-2.0
"""Rank-filtered interface proposals with residual-based adaptive regularization.

Only numerical force guesses are mixed. The physical solver owns its exact
acceptance test and transactional rollback. This uses the residual-reduction
ratio of Ouyang et al. (J. Sci. Comput. 96:5, 2023); its convergence assumptions
are not established for contact switches, so no convergence guarantee is made.
"""

import numpy as np


class SurfaceAnderson:
    """Short multisecant history, SVD filtering and measured-error safeguards."""

    def __init__(self, depth=6):
        self.depth = depth
        self.proposals = self.rejections = 0
        self.filtered_directions = 0
        self.reset()

    def reset(self):
        """Discard guesses whenever the underlying numerical force map changes."""
        self.history = []
        self.pending = None
        self.regularization = 1e-8
        self.last_ratio = None

    def propose(self, load, residual, relaxation):
        """Return an accelerated/recovery load, or None for ordinary Aitken.

        Every proposal needs a new physical evaluation from the same start.
        On inadequate measured reduction, return an Aitken step from the saved
        best iterate; neither a rejected trial nor its secant enters history.
        """
        load, residual = (
            np.asarray(load, dtype=float),
            np.asarray(residual, dtype=float),
        )
        if load.shape != residual.shape or load.ndim != 1:
            raise ValueError("Anderson loads and residuals must be matching vectors")
        if not np.isfinite(load).all() or not np.isfinite(residual).all() or not np.isfinite(relaxation):
            self.reset()
            return None
        norm = np.linalg.norm(residual)
        if self.pending is not None:
            reference, predicted, reference_max, fallback = self.pending
            self.pending = None
            ratio = (reference - norm) / max(reference - 0.95 * predicted, 1e-30)
            self.last_ratio = ratio
            if ratio < 0.1 or np.max(np.abs(residual)) > 1.05 * reference_max:
                self.history.clear()
                self.regularization = min(self.regularization * 10.0, 1e8)
                self.rejections += 1
                return fallback
            if ratio > 0.75:
                self.regularization = max(self.regularization * 0.1, 1e-12)
        self.history.append((load.copy(), residual.copy()))
        self.history = self.history[-self.depth - 1 :]
        if len(self.history) < 2 or norm <= 1e-30:
            return None

        norms = np.array([np.linalg.norm(item[1]) for item in self.history])
        # Anchor at the most recent best residual; the affine force combination
        # preserves action/reaction balance without component-wise clipping.
        best = len(norms) - 1 - int(np.argmin(norms[::-1]))
        saved_load, saved_residual = self.history[best]
        others = [item for index, item in enumerate(self.history) if index != best]
        # A contact interface often occupies only a few of the FEM nodes. Rows
        # exactly zero in every load and residual contribute nothing to either
        # the least-squares problem or its affine proposal. Keep every nonzero
        # value, including subnormal values: this is not a numerical cutoff.
        active = np.zeros(load.size, dtype=bool)
        for past_load, past_residual in self.history:
            active |= (past_load != 0) | (past_residual != 0)
        base_load, base_residual = saved_load[active], saved_residual[active]
        delta_load = np.column_stack([item[0][active] - base_load for item in others])
        delta_residual = np.column_stack([item[1][active] - base_residual for item in others])
        scale = np.max(np.linalg.norm(delta_residual, axis=0))
        if scale <= max(1e-30, norms[best] * 1e-12):
            return None
        try:
            left, singular, right = np.linalg.svd(delta_residual / scale, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        # A direct thin SVD avoids squaring the secant condition number. Drop
        # directions that are unresolved relative to the dominant secant.
        retained = singular > max(singular[0] * 1e-4, 1e-14)
        self.filtered_directions += int(np.count_nonzero(~retained))
        if not np.any(retained):
            return None
        ridge = self.regularization * (norms[best] / scale) ** 2
        weights = singular[retained] / (singular[retained] ** 2 + ridge)
        coefficients = -right[retained].T @ (weights * (left[:, retained].T @ (base_residual / scale)))
        if not np.isfinite(coefficients).all() or np.linalg.norm(coefficients, ord=1) > 64.0:
            return None
        predicted = base_residual + delta_residual @ coefficients
        predicted_norm = np.linalg.norm(predicted)
        if predicted_norm >= 0.95 * norms[best]:
            return None
        active_proposal = base_load + base_residual + (delta_load + delta_residual) @ coefficients
        step_scale = max(np.max(np.abs(base_load)), np.max(np.abs(base_residual)), 1e-30)
        if not np.isfinite(active_proposal).all() or np.max(np.abs(active_proposal - base_load)) > 2.0 * step_scale:
            return None

        # Nonmonotone reference: the best gets 55% weight and the other recent
        # iterates share 45%. The infinity guard matches the physical residual's
        # component-wise nature; physical convergence remains the caller's test.
        other_norms = np.delete(norms, best)
        reference = 0.55 * norms[best] + 0.45 * np.mean(other_norms)
        reference_max = 0.55 * np.max(np.abs(base_residual)) + 0.45 * np.mean(
            [np.max(np.abs(item[1])) for item in others]
        )
        self.pending = (
            reference,
            predicted_norm,
            reference_max,
            saved_load + relaxation * saved_residual,
        )
        proposal = np.zeros_like(load)
        proposal[active] = active_proposal
        self.proposals += 1
        return proposal
