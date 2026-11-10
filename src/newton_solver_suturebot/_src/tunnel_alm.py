# SPDX-License-Identifier: Apache-2.0
"""Augmented-Lagrangian contact with a barycentric material tunnel.

The virtual lumen has radial clearance and optional physical compliance.
Multipliers are force densities (N/m); track quadrature supplies their length.
Primal trials use the projected multipliers as forces. Only a converged shared
physical step may publish those multipliers as material history.
"""

import numpy as np
import warp as wp

from .insertion import _give_force, _shaft_projection


@wp.func
def _normal_projection(
    displacement: wp.vec3,
    tangent: wp.vec3,
    multiplier: wp.vec3,
    penalty: float,
    compliance: float,
    clearance: float,
):
    # Finite compliance has a unique displacement-only normal reaction.
    # Eliminating that dual exactly avoids redundant multiplier iterations;
    # zero compliance retains the hard-wall augmented-Lagrangian map.
    if compliance > 0.0:
        magnitude = wp.length(displacement)
        if magnitude > clearance:
            return (1.0 - clearance / magnitude) * displacement / compliance
        return wp.vec3(0.0)
    multiplier -= wp.dot(multiplier, tangent) * tangent
    augmented = multiplier + penalty * displacement
    magnitude = wp.length(augmented)
    result = wp.vec3(0.0)
    if magnitude > penalty * clearance:
        result = (1.0 - penalty * clearance / magnitude) * augmented / (1.0 + penalty * compliance)
    return result


@wp.kernel(enable_backward=False)
def _tunnel_forces(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    lengths: wp.array[float],
    engaged: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    fixed: wp.array[int],
    samples: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.vec3,
    body: int,
    normal: wp.array[wp.vec3],
    tangential: wp.array[float],
    next_normal: wp.array[wp.vec3],
    next_tangential: wp.array[float],
    penalty: float,
    compliance: float,
    clearance: float,
    tangent_radius: float,
    friction: float,
    dt: float,
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    diagnostics: wp.array[wp.vec3],
    statistics: wp.array[float],
):
    i = wp.tid()
    next_normal[i] = wp.vec3(0.0)
    next_tangential[i] = 0.0
    if engaged[i] == 0 or lengths[i] <= 0.0:
        return
    point, tissue_velocity = wp.vec3(0.0), wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
        tissue_velocity += bary[i][j] * qd[nodes[i][j]]
    pose = body_q[body]
    closest, tangent, distance, covered = _shaft_projection(point, pose, samples, tangent_radius)
    displacement = closest - point
    displacement -= wp.dot(displacement, tangent) * tangent
    previous_normal = normal[i] - wp.dot(normal[i], tangent) * tangent
    reaction = _normal_projection(displacement, tangent, previous_normal, penalty, compliance, clearance)
    if compliance > 0.0:
        # The normal constitutive equation is already satisfied exactly.
        # Only tangential history still contributes a dual fixed-point error.
        previous_normal = reaction
    center = wp.transform_point(pose, body_com)
    velocity = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), point - center)
    sliding = wp.dot(velocity - tissue_velocity, tangent)
    bound = friction * wp.length(reaction)
    drag = wp.clamp(tangential[i] + penalty * dt * sliding, -bound, bound)
    next_normal[i] = reaction
    next_tangential[i] = drag

    # The proximal normal residual enforces the normal cone of the clearance
    # disk. For positive reaction its fixed point is |d|-clearance=alpha*|lambda|.
    normal_error = wp.length(reaction - previous_normal) / penalty
    geometric_error = wp.max(0.0, wp.length(displacement) - clearance - compliance * wp.length(reaction))
    wp.atomic_max(statistics, 0, normal_error)
    wp.atomic_max(statistics, 1, geometric_error)
    # Sum changes before scatter: this bounds the resultant and any barycentric
    # nodal load change, including normal dual error as well as friction.
    wp.atomic_add(statistics, 2, lengths[i] * (wp.length(reaction - previous_normal) + wp.abs(drag - tangential[i])))
    scale = wp.max(wp.length(reaction), wp.length(previous_normal))
    scale = wp.max(scale, wp.max(wp.abs(drag), wp.abs(tangential[i])))
    wp.atomic_add(statistics, 3, lengths[i] * scale)
    if not wp.isfinite(reaction[0] + reaction[1] + reaction[2] + drag):
        wp.atomic_max(statistics, 4, 1.0)

    force = -lengths[i] * (reaction + drag * tangent)
    _give_force(force, point, center, body, body_f, diagnostics)
    for j in range(4):
        nodal = -bary[i][j] * force
        wp.atomic_add(particle_f, nodes[i][j], nodal)
        if fixed[nodes[i][j]] != 0:
            wp.atomic_add(diagnostics, 4, nodal)
            wp.atomic_add(diagnostics, 5, wp.cross(q[nodes[i][j]], nodal))
        else:
            wp.atomic_add(diagnostics, 2, nodal)
        wp.atomic_add(diagnostics, 3, wp.cross(q[nodes[i][j]], nodal))


@wp.kernel(enable_backward=False)
def _clear_unloaded(
    engaged: wp.array[int], lengths: wp.array[float], normal: wp.array[wp.vec3], tangential: wp.array[float]
):
    i = wp.tid()
    if engaged[i] == 0 or lengths[i] <= 0.0:
        normal[i] = wp.vec3(0.0)
        tangential[i] = 0.0


class _TunnelALM:
    """Trial-local proximal multipliers; the coupler owns the solve lifecycle."""

    def __init__(self, interaction, *, penalty_per_m, compliance_per_m, clearance, tolerance):
        parameters = (penalty_per_m, compliance_per_m, clearance, tolerance)
        if not np.isfinite(parameters).all() or min(parameters) < 0 or min(penalty_per_m, tolerance) <= 0:
            raise ValueError("Tunnel ALM needs positive penalty/tolerance and nonnegative compliance/clearance")
        self.interaction, self.device = interaction, interaction.device
        self.penalty_per_m, self.compliance_per_m = float(penalty_per_m), float(compliance_per_m)
        self.clearance, self.tolerance = float(clearance), float(tolerance)
        self.committed_normal = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self.committed_tangent = wp.zeros(0, dtype=float, device=self.device)
        self.normal = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self.tangent = wp.zeros(0, dtype=float, device=self.device)
        self.proposed_normal = wp.zeros(0, dtype=wp.vec3, device=self.device)
        self.proposed_tangent = wp.zeros(0, dtype=float, device=self.device)
        self._statistics = wp.zeros(5, dtype=float, device=self.device)
        self._open = False
        self.normal_error = self.dual_residual = 0.0
        self.dual_updates = self.total_dual_updates = 0

    def _resize(self):
        count = self.interaction.track_count
        old_count = self.committed_normal.size
        if count == old_count:
            return
        if count < old_count:
            raise RuntimeError("Reset tunnel history by constructing a new interaction")
        old_normal, old_tangent = self.committed_normal, self.committed_tangent
        self.committed_normal = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self.committed_tangent = wp.zeros(count, dtype=float, device=self.device)
        if old_count:
            wp.copy(self.committed_normal, old_normal, count=old_count)
            wp.copy(self.committed_tangent, old_tangent, count=old_count)
        self.normal = wp.empty_like(self.committed_normal)
        self.tangent = wp.empty_like(self.committed_tangent)
        self.proposed_normal = wp.empty_like(self.committed_normal)
        self.proposed_tangent = wp.empty_like(self.committed_tangent)

    def begin_step(self):
        if self._open:
            raise RuntimeError("Previous tunnel ALM step was neither committed nor discarded")
        self._resize()
        wp.copy(self.normal, self.committed_normal)
        wp.copy(self.tangent, self.committed_tangent)
        wp.copy(self.proposed_normal, self.committed_normal)
        wp.copy(self.proposed_tangent, self.committed_tangent)
        self._statistics.zero_()
        self.normal_error = self.dual_residual = 0.0
        self.dual_updates = 0
        self._open = True

    def add_forces(self, state, body_index, body_com_local, external_forces, dt):
        if not self._open:
            raise RuntimeError("Tunnel ALM requires a NeedleTissueCoupler solve, not explicit force exchange")
        interaction = self.interaction
        if self.normal.size != interaction.track_count:
            raise RuntimeError("Material tunnel history changed during an ALM trial")
        self._statistics.zero_()
        if interaction.track_count:
            wp.launch(
                _tunnel_forces,
                interaction.track_count,
                [
                    interaction._track_nodes,
                    interaction._track_bary,
                    interaction._track_lengths,
                    interaction._track_engaged,
                    state.particle_q,
                    state.particle_qd,
                    interaction._fixed,
                    interaction._samples,
                    state.body_q,
                    state.body_qd,
                    wp.vec3(*body_com_local),
                    body_index,
                    self.normal,
                    self.tangent,
                    self.proposed_normal,
                    self.proposed_tangent,
                    self.penalty_per_m,
                    self.compliance_per_m,
                    self.clearance,
                    float(interaction.radii.max()),
                    interaction.friction,
                    dt,
                    external_forces,
                    state.body_f,
                    interaction._diagnostics,
                    self._statistics,
                ],
                device=self.device,
            )

    def converged(self, absolute_tolerance, relative_tolerance):
        values = self._statistics.numpy()
        if values[4] or not np.isfinite(values).all():
            raise RuntimeError("Nonfinite tunnel ALM reaction or residual")
        self.normal_error = float(max(values[0], values[1]))
        self.dual_residual = float(values[2])
        return (
            self.normal_error <= self.tolerance
            and self.dual_residual <= absolute_tolerance + relative_tolerance * float(values[3])
        )

    def update_duals(self):
        if not self._open:
            raise RuntimeError("No tunnel ALM step is open")
        wp.copy(self.normal, self.proposed_normal)
        wp.copy(self.tangent, self.proposed_tangent)
        self.dual_updates += 1
        self.total_dual_updates += 1

    def commit(self):
        if not self._open:
            raise RuntimeError("No tunnel ALM step is open")
        wp.copy(self.committed_normal, self.proposed_normal)
        wp.copy(self.committed_tangent, self.proposed_tangent)
        # Acceptance can append a new anchor or release finite-shaft coverage.
        # Densities on unchanged IDs survive changing quadrature lengths.
        self._resize()
        interaction = self.interaction
        if interaction.track_count:
            wp.launch(
                _clear_unloaded,
                interaction.track_count,
                [interaction._track_engaged, interaction._track_lengths, self.committed_normal, self.committed_tangent],
                device=self.device,
            )
        self.discard()

    def discard(self):
        wp.copy(self.normal, self.committed_normal)
        wp.copy(self.tangent, self.committed_tangent)
        wp.copy(self.proposed_normal, self.committed_normal)
        wp.copy(self.proposed_tangent, self.committed_tangent)
        self._open = False
