# SPDX-License-Identifier: Apache-2.0
"""Optional native prepuncture search metric from the exact facet recorder.

D is a frozen positive search approximation, including friction majorization.
Use r-h J_full.T D J_free (M+h J_free.T D J_free)^-1 r, with exactly the
shaft preconditioner's FEM operator and condensed articulated mass. This is
only a search direction: original force, event, ALM and tet tests still decide
acceptance. Unmapped/fixed surface vertices have zero trial velocity; mapped
fixed vertices still receive the corresponding reaction in J_full.T.
"""

import numpy as np
import warp as wp

from .insertion import _surface_forces as _original_surface_kernel
from .shaft_preconditioner import (
    _native_needle_mass,
    _rigid_blocks_from_mass,
    _ShaftPreconditioner,
    base_diagonal,
    combine,
    invert_diagonal,
    prepare_rigid,
    unpack,
)
from .solvers.fem.differentiation import active_tape
from .subcycled_mujoco import SubcycledMuJoCo
from .surface_contact_tangent import SurfaceTangentAnchors


@wp.kernel(enable_backward=False)
def validate_recording(
    count: wp.array[int],
    flags: wp.array[int],
    nodes: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    particle_count: int,
    valid: wp.array[int],
):
    i = wp.tid()
    if i < count[0] and i < nodes.shape[0]:
        okay = True
        for j in range(3):
            okay = okay and nodes[i][j] >= -1 and nodes[i][j] < particle_count
            okay = okay and wp.isfinite(bary[i][j]) and bary[i][j] >= 0.0
            okay = okay and wp.isfinite(levers[i][j])
            for k in range(3):
                okay = okay and wp.isfinite(matrices[i][j, k])
        if not okay:
            wp.atomic_min(valid, 0, 0)
    if i == 0:
        if count[0] <= 0 or count[0] > nodes.shape[0] or flags[0] != 0 or flags[1] != 0:
            wp.atomic_min(valid, 0, 0)


@wp.kernel(enable_backward=False)
def pack_surface_statistics(
    statistics: wp.array[wp.float64],
    count: wp.array[int],
    valid: wp.array[int],
    result: wp.array[wp.float64],
):
    for i in range(6):
        result[i] = statistics[i]
    result[6] = wp.float64(count[0])
    result[7] = wp.float64(valid[0])


@wp.kernel(enable_backward=False)
def clear_direction_statistics(statistics: wp.array[wp.float64]):
    statistics[3] = wp.float64(0.0)
    statistics[4] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def surface_contact_action(
    fixed: wp.array[int],
    count: wp.array[int],
    valid: wp.array[int],
    nodes: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    x: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    if valid[0] == 0 or i >= count[0]:
        return
    n = fixed.shape[0]
    relative = x[n] + wp.cross(x[n + 1], levers[i])
    for j in range(3):
        node = nodes[i][j]
        if node >= 0:
            if fixed[node] == 0:
                relative -= bary[i][j] * x[node]
    force = h[0] * matrices[i] * relative
    for j in range(3):
        node = nodes[i][j]
        if node >= 0:
            if fixed[node] == 0:
                wp.atomic_add(out, node, -bary[i][j] * force)
    wp.atomic_add(out, n, force)
    wp.atomic_add(out, n + 1, wp.cross(levers[i], force))


@wp.kernel(enable_backward=False)
def surface_contact_diagonal(
    fixed: wp.array[int],
    count: wp.array[int],
    valid: wp.array[int],
    nodes: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    diagonal: wp.array[wp.mat33],
):
    i = wp.tid()
    if valid[0] == 0 or i >= count[0]:
        return
    matrix = h[0] * matrices[i]
    for j in range(3):
        node = nodes[i][j]
        if node >= 0:
            if fixed[node] == 0:
                wp.atomic_add(diagonal, node, bary[i][j] * bary[i][j] * matrix)
    n = fixed.shape[0]
    wp.atomic_add(diagonal, n, matrix)
    lever = levers[i]
    skew = wp.mat33(0.0, -lever[2], lever[1], lever[2], 0.0, -lever[0], -lever[1], lever[0], 0.0)
    wp.atomic_add(diagonal, n + 1, -skew * matrix * skew)


@wp.kernel(enable_backward=False)
def surface_balanced_correction(
    fixed: wp.array[int],
    count: wp.array[int],
    valid: wp.array[int],
    nodes: wp.array[wp.vec3i],
    bary: wp.array[wp.vec3],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    h: wp.array[float],
    x: wp.array[wp.vec3],
    output: wp.array[wp.float64],
):
    i = wp.tid()
    if valid[0] == 0 or i >= count[0]:
        return
    n = fixed.shape[0]
    relative = x[n] + wp.cross(x[n + 1], levers[i])
    for j in range(3):
        node = nodes[i][j]
        if node >= 0:
            if fixed[node] == 0:
                relative -= bary[i][j] * x[node]
    force = h[0] * matrices[i] * relative
    torque = wp.cross(levers[i], force)
    for j in range(3):
        node = nodes[i][j]
        if node >= 0:
            for k in range(3):
                wp.atomic_add(output, 3 * node + k, wp.float64(bary[i][j]) * wp.float64(force[k]))
    for k in range(3):
        wp.atomic_add(output, 3 * n + k, -wp.float64(force[k]))
        wp.atomic_add(output, 3 * (n + 1) + k, -wp.float64(torque[k]))


class SurfaceContactPreconditioner(_ShaftPreconditioner):
    """Fixed-capacity facet search metric, valid only for native first contact."""

    @staticmethod
    def supports(coupler):
        solver = coupler.rigid_solver
        return (
            type(solver) is SubcycledMuJoCo
            and getattr(solver.step, "__func__", None) is SubcycledMuJoCo.step
            and _ShaftPreconditioner.supports(coupler)
        )

    def __init__(self, coupler, capacity=4096):
        self.anchors = SurfaceTangentAnchors(capacity, coupler.model.device)
        self.valid = wp.zeros(1, dtype=int, device=coupler.model.device)
        self.result = wp.zeros(8, dtype=wp.float64, device=coupler.model.device)
        self.available = False
        super().__init__(coupler)
        self.count = capacity  # Launch capacity, never download the live count.
        self.matrices, self.levers = self.anchors.matrices, self.anchors.levers
        self.applications = 0
        self.selection_changes = 0
        self.unsupported_recordings = 0

    def begin_step(self):
        self._native_mass_ready = False
        self.active = False
        self.available = False
        self.valid.zero_()
        self.anchors.clear()
        self.coupler.interaction._surface_tangent_anchors = None

    def _recording_storage_supported(self):
        a = self.anchors
        specs = {
            "count": ((1,), wp.int32),
            "flags": ((2,), wp.int32),
            "nodes": ((self.count,), wp.vec3i),
            "bary": ((self.count,), wp.vec3),
            "points": ((self.count,), wp.vec3),
            "levers": ((self.count,), wp.vec3),
            "normals": ((self.count,), wp.vec3),
            "tangents": ((self.count,), wp.vec3),
            "matrices": ((self.count,), wp.mat33),
            "sources": ((self.count,), wp.vec2i),
            "scalars": ((self.count, 7), wp.float32),
        }
        return all(
            isinstance(value := getattr(a, name, None), wp.array)
            and value.shape == shape
            and value.dtype == dtype
            and value.device == self.device
            and not value.requires_grad
            for name, (shape, dtype) in specs.items()
        ) and all(not value.requires_grad for value in (self.valid, self.result))

    def begin_trials(self, state, control):
        c = self.coupler
        from . import insertion, needle_trial_forces
        from .coupling import NeedleTissueCoupler
        from .insertion import NeedleTissueInteraction

        law = np.asarray(
            [c.interaction.stiffness, c.interaction.damping, c.interaction.friction, c.interaction.speed_scale]
        )
        self.available = bool(
            type(c.interaction) is NeedleTissueInteraction
            and insertion._surface_forces is _original_surface_kernel
            and needle_trial_forces._surface_forces is _original_surface_kernel
            and self._recording_storage_supported()
            and all(
                getattr(getattr(c.interaction, name), "__func__", None) is getattr(NeedleTissueInteraction, name)
                for name in ("add_forces", "_add_forces", "_visited_sample_mask", "_tip_boundary_permission")
            )
            and all(
                getattr(getattr(c, name), "__func__", None) is getattr(NeedleTissueCoupler, name)
                for name in ("_step_fem", "_evaluate")
            )
            and np.isfinite(law).all()
            and np.all(law >= 0)
            and law[3] > 0
            and not c.interaction.punctured
            and c.interaction.track_count == 0
            and not getattr(c.model, "requires_grad", False)
            and not getattr(c.fem_solver, "differentiable", False)
            and active_tape() is None
            and not any(
                array is not None and array.requires_grad
                for value in (state, c._candidate)
                for array in (
                    value.particle_q,
                    value.particle_qd,
                    value.particle_f,
                    value.joint_q,
                    value.joint_qd,
                    value.body_q,
                    value.body_qd,
                    value.body_f,
                )
            )
            and not any(
                array is not None and array.requires_grad
                for array in (
                    getattr(control, "joint_f", None),
                    getattr(control, "joint_target_q", None),
                    getattr(control, "joint_target_qd", None),
                    getattr(control, "tet_activations", None),
                    getattr(getattr(control, "mujoco", None), "ctrl", None),
                )
            )
        )
        c.interaction._surface_tangent_anchors = self.anchors if self.available else None
        c.interaction._surface_tangent_moment_length = c.moment_length
        return self.available

    def validate(self):
        self.valid.fill_(1)
        a = self.anchors
        wp.launch(
            validate_recording,
            max(self.count, 1),
            [a.count, a.flags, a.nodes, a.bary, a.matrices, a.levers, self.n - 2, self.valid],
            device=self.device,
        )

    def pack_statistics(self, statistics):
        """Append validity to the existing readback, without creating a fence."""
        self.validate()
        wp.launch(
            pack_surface_statistics, 1, [statistics, self.anchors.count, self.valid, self.result], device=self.device
        )
        return self.result

    def _contact_arguments(self):
        a = self.anchors
        return [self.fixed, a.count, self.valid, a.nodes, a.bary, a.matrices, a.levers, self.time_step]

    def matvec(self, x, y, z, alpha, beta):
        self.mass_apply(x, self.work)
        wp.launch(surface_contact_action, self.count, self._contact_arguments() + [x, self.work], device=self.device)
        wp.launch(combine, self.n, [self.work, y, z, alpha, beta], device=self.device)

    def _solve(self, residual, output):
        self.z.zero_()
        self.linear_solver(b=self.rhs, x=self.z, M=self.inverse)
        wp.copy(output, residual)
        wp.launch(
            surface_balanced_correction, self.count, self._contact_arguments() + [self.z, output], device=self.device
        )

    def apply(self, state, dt, residual, output):
        if not self.available or not self.active or active_tape() is not None:
            wp.copy(output, residual)
            return False
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Surface preconditioner timestep must be finite and positive")
        c = self.coupler
        if self._last_dt != dt:
            self.time_step.assign(np.asarray([dt], dtype=np.float32))
            self._last_dt = dt
        damping = np.asarray(c.rigid_solver.mj_model.dof_damping[:6], dtype=np.float32)
        if not np.isfinite(damping).all() or np.any(damping < 0):
            wp.copy(output, residual)
            return False
        if self._last_damping is None or not np.array_equal(damping, self._last_damping):
            self.native_damping.assign(damping)
            self._last_damping = damping.copy()
        if self._articulated and not self._native_mass_ready:
            try:
                metric = _native_needle_mass(
                    c.rigid_solver.mj_model, c.rigid_solver.mj_data, 1, c.moment_length, dt, self._native_scratch
                )
                self.rigid_blocks.assign(_rigid_blocks_from_mass(metric))
                self._native_mass_ready = True
            except (ValueError, np.linalg.LinAlgError):
                self.native_mass_fallbacks += 1
                wp.copy(output, residual)
                return False
        if not self._articulated:
            wp.launch(
                prepare_rigid,
                1,
                [
                    state.body_q,
                    self.mass,
                    self.inertia,
                    self.native_damping,
                    self.time_step,
                    c.moment_length,
                    self.rigid_blocks,
                ],
                device=self.device,
            )
        matrix = c.fem_solver.A_bsr
        wp.launch(
            base_diagonal,
            self.n,
            [self.fixed, matrix.offsets, matrix.columns, matrix.values, self.rigid_blocks, self.diagonal],
            device=self.device,
        )
        wp.launch(surface_contact_diagonal, self.count, self._contact_arguments() + [self.diagonal], device=self.device)
        wp.launch(invert_diagonal, self.n, [self.diagonal, self.inverse_diagonal], device=self.device)
        wp.launch(unpack, self.n, [residual, self.rhs], device=self.device)
        fixed = c.fem_solver._fixed_particles
        projected = None if fixed is None else fixed._projected
        arrays = (
            matrix.offsets,
            matrix.columns,
            matrix.values,
            *self._contact_arguments(),
            residual,
            output,
            *(() if projected is None else (projected.values,)),
        )
        key = tuple((a.ptr, a.shape, a.dtype) for a in arrays)
        if not self._conditional:
            self._solve(residual, output)
        elif key != self._graph_key:
            self._graph = None
            self._graph_key = key
            self._graph_inputs = arrays
            self._solve(residual, output)
        else:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._solve(residual, output)
                self._graph = capture.graph
                self.graph_captures += 1
            wp.capture_launch(self._graph)
            self.graph_launches += 1
        self.applications += 1
        return True
