# SPDX-License-Identifier: Apache-2.0
"""Captured simultaneous motion/friction solve using prepared FEM mechanics."""

import warp as wp
from warp.optim.linear import LinearOperator, gmres

from ...coupling import _apply_interface_load, _interface_residual
from ...mujoco_free_body_gpu import MuJoCoFreeBodyGPU
from ...needle_trial_forces import _NeedleTrialForces
from . import control as controller
from . import kernels as k


class _SimultaneousTrials:
    def __init__(self, solver):
        self.c = solver
        self.device = solver.model.device
        self.rigid = MuJoCoFreeBodyGPU(solver.model, solver.rigid_solver)
        self.permissions = _NeedleTrialForces(
            solver.interaction, query_block_dim=getattr(solver, "_needle_query_block_dim", None)
        )
        self.state = wp.zeros(6, dtype=int, device=self.device)
        self.merit = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.merit_weights = wp.ones(3, dtype=wp.float64, device=self.device)
        self.running = wp.zeros(1, dtype=int, device=self.device)
        self.continuing = wp.zeros(1, dtype=int, device=self.device)
        self.searching = wp.zeros(1, dtype=int, device=self.device)
        self.finite = wp.ones(1, dtype=int, device=self.device)
        self.step_length = wp.ones(1, dtype=float, device=self.device)
        self.result = wp.zeros(11, dtype=wp.float64, device=self.device)
        self.final_corrections = wp.zeros(1, dtype=int, device=self.device)
        self.stats = wp.zeros(6, dtype=wp.float64, device=self.device)
        # Squared impulse residual, squared RHS norm, nonfinite flag.
        # Applied-load relative accuracy is authoritative. Fresh-law relative
        # accuracy remains diagnostic; its force-unit defect is still checked.
        self.fem_statistics = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.fresh_fem_statistics = wp.zeros(3, dtype=wp.float64, device=self.device)
        self.momentum_error = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.blocks = wp.zeros(3, dtype=wp.mat33, device=self.device)
        self.rigid_rhs = wp.zeros(2, dtype=wp.vec3, device=self.device)
        self.signature = None
        self.graphs = {}
        self.captures = self.launches = 0

    def invalidate(self):
        self.graphs.clear()
        self.signature = None

    def _launch(self, kernel, count, args):
        wp.launch(kernel, count, args, device=self.device)

    def prepare(self, state_in, control, dt):
        c = self.c
        x = c.interaction
        self.rigid.prepare(dt)
        self.permissions.prepare(x)
        self.prepared = c.fem_solver.prepare_step(state_in, control, None, dt)
        self.linear_settings = self.prepared.prepare_solve()
        self.dt = dt
        self.n, self.m = c.model.particle_count, x.track_count
        signature = (
            dt,
            self.n,
            self.m,
            id(self.prepared.operator),
            id(c.fem_solver.M_bsr),
            self.prepared.unconstrained_matrix.values.ptr,
            self.permissions.signature(),
            x.tunnel_alm.penalty_per_m,
            x.tunnel_alm.compliance_per_m,
            x.tunnel_alm.clearance,
            x.tunnel_alm.tolerance,
            x.friction,
            c.moment_length,
            c.max_iterations,
            c.absolute_tolerance,
            c.relative_tolerance,
            self.linear_settings,
        )
        if signature != self.signature:
            self.graphs.clear()
            self.signature = signature
            size = self.n + 2 + self.m

            def vector(count=size):
                return wp.zeros(count, dtype=wp.vec3, device=self.device)

            self.unknown, self.base, self.direction, self.residual, self.rhs = [vector() for _ in range(5)]
            self.work, self.pc_work, self.pc_correction = [vector() for _ in range(3)]
            self.action = vector(self.n)
            self.certification_rhs = vector(self.n)
            self.certification_load = vector(self.n)
            self.tissue_increment = self.unknown[: self.n]
            self.diagonal = wp.zeros(self.n + 2, dtype=wp.mat33, device=self.device)
            self.inverse = wp.zeros_like(self.diagonal)
            self.normal_matrix = wp.zeros(self.m, dtype=wp.mat33, device=self.device)
            self.tangent, self.normal_direction, self.lever, self.geometric_lever, self.coefficients = [
                vector(self.m) for _ in range(5)
            ]
            self.operator = LinearOperator((size, size), wp.vec3, self.device, self.matvec)
            self.preconditioner = LinearOperator((size, size), wp.vec3, self.device, self.precondition)
            self.linear = gmres(
                self.operator,
                self.rhs,
                self.direction,
                M=self.preconditioner,
                restart=16,
                tol=0.02,
                atol=0.0,
                maxiter=128,
                check_every=0,
                use_cuda_graph=True,
                run=False,
            )

    def matvec(self, x, y, z, alpha, beta):
        self.prepared.apply_operator(x[: self.n], self.action)
        self._launch(k.base_action, len(x), [x, self.action, self.blocks, self.prepared.fixed_mask, self.dt, self.work])
        a = self.c.interaction
        self._launch(
            k.contact_action,
            self.m,
            [
                x,
                self.prepared.fixed_mask,
                a._track_nodes,
                a._track_bary,
                self.normal_matrix,
                self.tangent,
                self.normal_direction,
                self.lever,
                self.geometric_lever,
                self.coefficients,
                self.work,
                0.01,
            ],
        )
        self._launch(k.combine, len(x), [self.work, y, z, alpha, beta])

    def precondition(self, x, y, z, alpha, beta):
        self.pc_correction.zero_()
        self._launch(k.precondition_start, self.n + 2, [x, self.inverse, self.pc_work])
        a = self.c.interaction
        self._launch(
            k.precondition_friction,
            self.m,
            [
                x,
                self.pc_work,
                self.inverse,
                self.prepared.fixed_mask,
                a._track_nodes,
                a._track_bary,
                self.normal_matrix,
                self.tangent,
                self.normal_direction,
                self.lever,
                self.geometric_lever,
                self.coefficients,
                self.pc_correction,
                self.state[4:5],
                0.01,
            ],
        )
        self._launch(k.precondition_finish, self.n + 2, [self.pc_work, self.inverse, self.pc_correction])
        self._launch(k.combine, len(x), [self.pc_work, y, z, alpha, beta])

    def linearize(self):
        c, x = self.c, self.c.interaction
        a = x.tunnel_alm
        state = c._candidate
        self._launch(
            k.linearize_anchors,
            self.m,
            [
                x._track_nodes,
                x._track_bary,
                x._track_lengths,
                x._track_engaged,
                state.particle_q,
                state.particle_qd,
                x._samples,
                state.body_q,
                state.body_qd,
                a.tangent,
                float(x.radii.max()),
                a.clearance,
                a.compliance_per_m,
                a.penalty_per_m,
                x.friction,
                self.dt,
                c.moment_length,
                self.normal_matrix,
                self.tangent,
                self.normal_direction,
                self.lever,
                self.geometric_lever,
                self.coefficients,
            ],
        )
        self._launch(
            k.base_diagonal,
            self.n + 2,
            [self.prepared.diagonal_blocks, self.prepared.fixed_mask, self.blocks, self.dt, self.diagonal],
        )
        self._launch(
            k.contact_diagonal,
            self.m,
            [
                self.prepared.fixed_mask,
                x._track_nodes,
                x._track_bary,
                self.normal_matrix,
                self.tangent,
                self.lever,
                self.coefficients,
                self.diagonal,
            ],
        )
        self._launch(k.invert, self.n + 2, [self.diagonal, self.inverse])

    def forces(self):
        c = self.c
        c._forces.zero_()
        c._candidate.body_f.zero_()
        self.permissions.add_forces(c._candidate, c.needle_body, c.needle_com, c._forces, self.dt)

    def evaluate(self, state_in, *, predict=False):
        c = self.c
        a = c.interaction.tunnel_alm
        self.prepared.reconstruct(self.tissue_increment, c._candidate)
        self._launch(
            k.reconstruct_rigid,
            1,
            [
                state_in.joint_q,
                state_in.joint_qd,
                self.unknown,
                self.n,
                self.dt,
                c.moment_length,
                c._candidate.joint_q,
                c._candidate.joint_qd,
                c._candidate.body_q,
                c._candidate.body_qd,
            ],
        )
        self._launch(k.set_tangent, self.m, [self.unknown, self.n, a.penalty_per_m * self.dt, a.tangent])
        self.check_finite()
        wp.capture_if(self.finite, lambda: self.evaluate_finite(predict=predict))
        self._launch(controller.invalid_trial, 1, [self.finite, self.state, self.merit])

    def check_finite(self):
        c = self.c
        # Both Newton proposals and canonical certification endpoints must be
        # checked before entering geometry queries.
        self.finite.fill_(1)
        for values in (self.unknown, c._candidate.particle_q, c._candidate.particle_qd):
            self._launch(controller.finite_vectors, len(values), [values, self.finite])
        for values in (c._candidate.joint_q, c._candidate.joint_qd):
            self._launch(controller.finite_scalars, len(values), [values, self.finite])

    def evaluate_finite(self, *, predict=False):
        c = self.c
        a = c.interaction.tunnel_alm
        x = c.interaction
        if predict:
            self._launch(
                k.predict_tangent,
                self.m,
                [
                    self.unknown,
                    x._track_nodes,
                    x._track_bary,
                    x._track_engaged,
                    c._candidate.particle_q,
                    c._candidate.particle_qd,
                    x._samples,
                    c._candidate.body_q,
                    c._candidate.body_qd,
                    float(x.radii.max()),
                    a.clearance,
                    a.compliance_per_m,
                    x.friction,
                    a.penalty_per_m * self.dt,
                    a.tangent,
                ],
            )
        self.forces()
        self._launch(
            k.combine,
            self.n,
            [c._forces, c._external_nodes, self.certification_load, 1.0, 1.0],
        )
        self.prepared.rhs(self.certification_load, out=self.certification_rhs)
        self.prepared.apply_operator(self.tissue_increment, self.action)
        self.stats.zero_()
        self.fem_statistics.zero_()
        self._launch(
            k.momentum_residual,
            self.n + 2,
            [
                self.action,
                self.certification_rhs,
                c._forces,
                c._candidate.body_f,
                self.prepared.fixed_mask,
                self.unknown,
                self.blocks,
                self.rigid_rhs,
                self.dt,
                c.moment_length,
                self.residual,
                self.stats,
                self.fem_statistics,
            ],
        )
        self._launch(
            k.friction_residual,
            self.m,
            [
                self.unknown,
                self.n,
                a.penalty_per_m * self.dt,
                c.interaction._track_lengths,
                c.interaction._track_engaged,
                a.tangent,
                a.proposed_tangent,
                self.residual,
                self.stats,
            ],
        )
        self._launch(
            controller.measure,
            1,
            [
                self.stats,
                a._statistics,
                self.fem_statistics,
                self.permissions.fallback,
                c.absolute_tolerance,
                c.relative_tolerance,
                a.tolerance,
                c.fem_solver.linear_solver_tolerance,
                self.dt,
                self.merit_weights,
                self.state,
                self.merit,
            ],
        )

    def interface_residual(self):
        c = self.c
        c._statistics.zero_()
        self._launch(
            _interface_residual,
            len(c._load),
            [
                c._forces,
                c._candidate.body_f,
                c.needle_body,
                c.moment_length,
                c._trial_load,
                c._previous_residual,
                0,
                c._value,
                c._residual,
                c._statistics,
            ],
        )

    def solve(self, state_in, control, dt):
        self.prepare(state_in, control, dt)
        c = self.c
        a = c.interaction.tunnel_alm

        def apply_load():
            self._launch(
                _apply_interface_load,
                max(self.n, c.model.body_count),
                [
                    c._trial_load,
                    c._external_nodes,
                    c._external_wrench,
                    c.needle_body,
                    c.moment_length,
                    c._forces,
                    state_in.body_f,
                ],
            )

        def decide():
            self._launch(
                controller.decide,
                1,
                [self.state, self.merit, self.permissions.fallback, c.max_iterations - 1, self.running],
            )

        def search():
            self._launch(k.proposal, len(self.unknown), [self.base, self.direction, self.step_length, self.unknown])
            self.evaluate(state_in)
            self._launch(
                controller.assess_search,
                1,
                [self.state, self.merit, self.step_length, self.searching, c.max_iterations - 1],
            )

        def iteration():
            self.linearize()
            self.direction.zero_()
            self._launch(k.negative, len(self.unknown), [self.residual, self.rhs])
            self.linear(b=self.rhs, x=self.direction, M=self.preconditioner)
            wp.copy(self.base, self.unknown)
            self._launch(
                controller.begin_search,
                1,
                [
                    self.state,
                    self.merit,
                    self.step_length,
                    self.searching,
                    self.stats,
                    a._statistics,
                    self.fem_statistics,
                    c.absolute_tolerance,
                    c.relative_tolerance,
                    c.fem_solver.linear_solver_tolerance,
                    dt,
                    self.merit_weights,
                ],
            )
            wp.capture_while(self.searching, search)
            decide()

        def certify_finite():
            self.forces()
            self.interface_residual()
            self.momentum_error.zero_()
            self.fem_statistics.zero_()
            self.fresh_fem_statistics.zero_()
            self._launch(
                k.momentum_defect,
                self.n,
                [
                    self.action,
                    self.certification_rhs,
                    self.prepared.fixed_mask,
                    dt,
                    self.momentum_error,
                    c._statistics,
                    self.fem_statistics,
                ],
            )
            # Check the fresh law too: two separately bounded defects would
            # otherwise permit up to twice the force tolerance in momentum.
            self._launch(
                k.combine,
                self.n,
                [c._forces, c._external_nodes, self.certification_load, 1.0, 1.0],
            )
            self.prepared.rhs(self.certification_load, out=self.certification_rhs)
            self._launch(
                k.momentum_defect,
                self.n,
                [
                    self.action,
                    self.certification_rhs,
                    self.prepared.fixed_mask,
                    dt,
                    self.momentum_error,
                    c._statistics,
                    self.fresh_fem_statistics,
                ],
            )
            self._launch(
                controller.certification,
                1,
                [
                    c._statistics,
                    a._statistics,
                    self.fem_statistics,
                    self.fresh_fem_statistics,
                    self.permissions.fallback,
                    c.absolute_tolerance,
                    c.relative_tolerance,
                    a.tolerance,
                    c.fem_solver.linear_solver_tolerance,
                    self.state,
                    self.merit,
                ],
            )

        def certify():
            # Repair only the proposed endpoint under the balanced applied
            # law L0. The joint increment is the numerical warm start; no
            # physical state is advanced from an intermediate endpoint.
            # One affine solve runs per certification, never per Newton trial.
            self._launch(controller.record_correction, 1, [self.final_corrections])
            self.interface_residual()
            wp.copy(c._trial_load, c._value)
            apply_load()
            self.prepared.rhs(c._forces, out=self.certification_rhs)
            wp.copy(self.prepared.velocity_increment, self.tissue_increment)
            self.prepared.solve(self.certification_rhs, out=self.tissue_increment)
            self.prepared.apply_operator(self.tissue_increment, self.action)
            self.prepared.reconstruct(self.tissue_increment, c._candidate)
            # The rigid load remains the current physical wrench. Its canonical
            # integration is retained for the bitwise native endpoint audit.
            self.rigid.step(state_in, c._candidate, control, dt)
            self.check_finite()
            wp.capture_if(self.finite, certify_finite)
            self._launch(controller.invalid_trial, 1, [self.finite, self.state, self.merit])
            wp.copy(c.fem_solver.velocity_increment, self.tissue_increment)
            self._launch(
                controller.continue_after_certificate,
                1,
                [self.state, self.merit, self.permissions.fallback, c.max_iterations - 1, self.continuing],
            )

        def recover():
            # Keep the corrected tissue increment and tau; restart the rigid
            # increment at the canonical endpoint before full reevaluation.
            # Another correction may be attempted within the shared budget.
            self._launch(
                k.recover_rigid_increment,
                1,
                [
                    state_in.joint_q,
                    state_in.joint_qd,
                    c._candidate.joint_q,
                    c._candidate.joint_qd,
                    self.n,
                    c.moment_length,
                    self.unknown,
                ],
            )
            self.evaluate(state_in)
            decide()

        def cycle():
            self.continuing.zero_()
            wp.capture_while(self.running, iteration)
            wp.capture_if(self.state[3:4], certify)
            wp.capture_if(self.continuing, recover)

        inputs = (
            state_in.particle_q,
            state_in.particle_qd,
            state_in.joint_q,
            state_in.joint_qd,
            state_in.body_f,
            None if control is None else control.joint_f,
        )
        key = tuple(None if v is None else (v.ptr, v.shape) for v in inputs)
        if key not in self.graphs:
            with wp.ScopedCapture(device=self.device) as capture:
                self._launch(controller.initialize, 1, [self.state, self.merit, self.running, self.merit_weights])
                self.momentum_error.zero_()
                self.final_corrections.zero_()
                c._candidate.assign(state_in)
                joint = self.rigid.zero_joint_force if control is None else control.joint_f
                self._launch(
                    k.rigid_system,
                    1,
                    [
                        state_in.joint_q,
                        state_in.joint_qd,
                        c._external_wrench,
                        joint,
                        self.rigid.mass,
                        self.rigid.inertia,
                        self.rigid.damping,
                        dt,
                        c.moment_length,
                        self.blocks,
                        self.rigid_rhs,
                    ],
                )
                # The simultaneous solve corrects momentum directly, so its
                # initial guess can reuse the accepted tissue increment.
                self._launch(
                    k.initialize,
                    len(self.unknown),
                    [
                        self.prepared.velocity_increment,
                        c._trial_load,
                        a.tangent,
                        self.blocks,
                        self.rigid_rhs,
                        a.penalty_per_m * dt,
                        self.n,
                        self.unknown,
                    ],
                )
                self.evaluate(state_in, predict=True)
                decide()
                self.continuing.fill_(1)
                wp.capture_while(self.continuing, cycle)
                self._launch(controller.result, 1, [self.state, self.merit, self.permissions.fallback, self.result])
            if len(self.graphs) >= 2:
                self.graphs.clear()
            self.graphs[key] = (capture.graph, inputs)
            self.captures += 1
        wp.capture_launch(self.graphs[key][0])
        self.launches += 1
        self.last_result = self.result.numpy()
        return self.last_result
