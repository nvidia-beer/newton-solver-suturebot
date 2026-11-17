# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The correction's anchor tangent holds the shear of a sliding tunnel anchor, and its solve is not symmetric.

On the sliding branch of the tunnel the drag is ``friction * |wall reaction|`` along the shaft. The anchor
force then changes with the reaction by ``(I + S) dR``, ``S = friction * sign * t d.T`` (``t`` the shaft
tangent, ``d`` the unit direction of the reaction). The anchor kernels write ``S`` next to the tangent
``D``; the kernels that apply the anchors apply ``(I + S) D``; the solve of that operator is BiCGSTAB.
Force law, residual, tolerance and trial budget are the same as before: only the search direction changes.

Checked here:

- ``D`` and ``S`` against a central difference of the unchanged force kernel, sliding and sticking;
- operator, diagonal and balanced correction against a dense oracle of ``M + h J.T (I + S) D J``;
- the solver follows the anchor law: BiCGSTAB with a friction tunnel (whatever the friction coefficient),
  CG for the penalty tunnel, without a tunnel, for the surface-only metric and for a track without samples;
- both solvers are made with the correction, whatever law its track follows then: a tunnel model attached
  or removed after construction (no scene does it) makes and releases neither a solver nor an anchor
  array, and the captured trial loop captures anew;
- without a shear every changed kernel writes the bits of its earlier text, which is rebuilt from the
  kernel's source by undoing the edit, and a whole correction of the penalty tunnel holds the same bits;
- a breakdown of BiCGSTAB (a division by zero in the stock solver) gives a non-finite correction, and
  the trial that applies it is rejected by the tests the trial loops already have, with no guard in between;
- one captured correction replays at changing numbers of track samples, and takes the solver of each
  launch by the live track count on the device;
- the solver's kernels are created and loaded with the solver, not by the first correction.
"""

import inspect
import linecache
import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_needle_coupling_graph as coupled
import test_woodbury_device as woodbury
import warp as wp
from _woodbury_reference import block_diagonal, interpolation, rigid_interpolation
from test_combined_contact_preconditioner import native_embedded
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import _apply, dense_action, history, setup
from test_needle_tissue_coupler import _native_state
from warp.optim.linear import CG, BiCGSTAB, LinearOperator, bicgstab, cg

from newton_solver_suturebot import NeedleCouplingNotConverged, NeedleStepTooLarge, NeedleTissueCoupler
from newton_solver_suturebot._src.solvers.coupled.acceleration import shaft_preconditioner
from newton_solver_suturebot._src.solvers.coupled.acceleration.combined_contact_preconditioner import (
    CombinedContactPreconditioner,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import (
    _ShaftPreconditioner,
    balanced_correction,
    base_diagonal,
    combine,
    contact_action,
    contact_diagonal,
    invert_diagonal,
    prepare_alm_anchors,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_preconditioner import (
    SurfaceContactPreconditioner,
)
from newton_solver_suturebot._src.solvers.coupled.acceleration.woodbury_guess import WoodburyInitialGuess
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _tunnel_forces, _TunnelALM
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
H = woodbury.H
FRICTION = 0.2


# ---------------------------------------------------------------------------------------------
# One track sample next to a straight shaft: the force kernel and the anchor kernel on the same state.


class Sample:
    """A tissue point 0.1 to 0.2 mm off a straight shaft, well outside the clearance of the tunnel."""

    PENALTY, CLEARANCE, RADIUS, LENGTH = 60000.0, 2.0e-5, 0.0005, 0.00025

    def __init__(self, seed, *, sliding_speed=0.0, stick=False, compliance=1.0 / 300000.0, engaged=1, length=None):
        rng = np.random.default_rng(seed)
        self.compliance = compliance
        line = np.stack([np.linspace(-0.01, 0.01, 21), np.zeros(21), np.zeros(21)], axis=1)
        self.samples = self.array(line, wp.vec3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.2, 1.2)
        self.quat = np.r_[np.sin(angle / 2) * axis, np.cos(angle / 2)]
        self.position = rng.normal(0, 0.01, 3)
        rotation = np.array(wp.quat_to_matrix(wp.quat(*self.quat))).reshape(3, 3)
        self.tangent = rotation[:, 0]
        normal = np.cross(self.tangent, rng.normal(size=3))
        normal /= np.linalg.norm(normal)
        point = self.position + rotation @ np.array([rng.uniform(-0.004, 0.004), 0, 0])
        point = point + rng.uniform(1.0e-4, 2.0e-4) * normal
        tet = point + rng.normal(0, 0.001, (4, 3))
        weights = np.linalg.lstsq(np.vstack([tet.T, np.ones(4)]), np.r_[point, 1.0], rcond=None)[0]
        self.q = self.array(tet, wp.vec3)
        self.qd = self.array(np.zeros((4, 3)), wp.vec3)
        self.nodes = wp.array(np.array([[0, 1, 2, 3]], np.int32), dtype=wp.vec4i, device=DEVICE)
        self.bary = self.array(weights[None], wp.vec4)
        self.lengths = self.array([self.LENGTH if length is None else length], float)
        self.engaged = wp.array(np.array([engaged], np.int32), dtype=int, device=DEVICE)
        # Live counts of an interaction [visited, track, openings]: the anchor kernel runs the one sample.
        self.live = wp.array(np.array([0, 1, 0], np.int32), dtype=int, device=DEVICE)
        self.fixed = wp.zeros(4, dtype=int, device=DEVICE)
        self.velocity = sliding_speed * self.tangent + rng.normal(0, 2.0e-4, 3)
        # A sample that has been sliding holds a drag multiplier at its bound (below 11 N/m here): the trial
        # value, multiplier + penalty * dt * sliding, lies beyond the bound of the trial for more sliding that way.
        held_drag = np.sign(sliding_speed) * 12.0
        if stick:
            # A drag multiplier inside the bound and almost no sliding.
            held_drag = 0.0
            self.velocity = rng.normal(0, 1.0e-5, 3)
        self.multiplier_normal = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        self.multiplier_drag = self.array([held_drag], float)
        self.dt = self.array([H], float)

    @staticmethod
    def array(values, dtype):
        return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device=DEVICE)

    def state(self, dv):
        """The needle after an implicit step whose end velocity is changed by ``dv``."""
        pose = np.r_[self.position + H * dv, self.quat]
        return (
            self.array(pose[None], wp.transform),
            self.array(np.r_[self.velocity + dv, np.zeros(3)][None], wp.spatial_vector),
        )

    def force(self, dv):
        """Force of the unchanged tunnel kernel on the needle [N]."""
        body_q, body_qd = self.state(dv)
        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE)
        wp.launch(
            _tunnel_forces,
            1,
            [
                self.nodes,
                self.bary,
                self.lengths,
                self.engaged,
                self.q,
                self.qd,
                self.fixed,
                self.samples,
                body_q,
                body_qd,
                wp.vec3(0.0),
                0,
                self.multiplier_normal,
                self.multiplier_drag,
                wp.zeros(1, dtype=wp.vec3, device=DEVICE),
                wp.zeros(1, dtype=float, device=DEVICE),
                self.PENALTY,
                self.compliance,
                self.CLEARANCE,
                self.RADIUS,
                FRICTION,
                self.dt,
                wp.zeros(4, dtype=wp.vec3, device=DEVICE),
                body_f,
                wp.zeros(6, dtype=wp.vec3, device=DEVICE),
                wp.zeros(5, dtype=float, device=DEVICE),
            ],
            device=DEVICE,
        )
        return body_f.numpy()[0, :3].astype(float)

    def anchor_inputs(self, friction=FRICTION):
        """Arguments of ``prepare_alm_anchors`` up to its outputs."""
        body_q, body_qd = self.state(np.zeros(3))
        return [
            self.nodes,
            self.bary,
            self.lengths,
            self.engaged,
            self.live,
            self.q,
            self.qd,
            self.samples,
            body_q,
            body_qd,
            0,
            wp.vec3(0.0),
            self.RADIUS,
            self.multiplier_normal,
            self.multiplier_drag,
            self.PENALTY,
            self.compliance,
            self.CLEARANCE,
            friction,
            self.dt,
            0.01,
        ]

    def anchor(self, friction=FRICTION):
        """Tangent ``D`` and shear ``S`` of the correction for this sample."""
        matrices = wp.zeros(1, dtype=wp.mat33, device=DEVICE)
        levers = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        # Start from a value the kernel does not write: a sample without a shear must be set to zero.
        shears = wp.full(1, wp.mat33(*([7.0] * 9)), dtype=wp.mat33, device=DEVICE)
        wp.launch(prepare_alm_anchors, 1, [*self.anchor_inputs(friction), matrices, levers, shears], device=DEVICE)
        return matrices.numpy()[0].astype(float), shears.numpy()[0].astype(float)

    def measured(self):
        """Central difference of the needle force over the end velocity; the needle moves by 2 um."""
        step = 2.0e-6 / H
        result = np.zeros((3, 3))
        for axis in range(3):
            dv = np.zeros(3)
            dv[axis] = step
            result[:, axis] = (self.force(dv) - self.force(-dv)) / (2 * step)
        return result


class AnchorTangentTests(unittest.TestCase):
    def test_tangent_with_the_shear_is_the_derivative_of_the_tunnel_force(self):
        # Compliant wall (the grasp scene) and hard wall, four samples each.
        for compliance in (1.0 / 300000.0, 0.0):
            for seed in range(4):
                for speed in (0.005, -0.005):
                    with self.subTest(compliance=compliance, seed=seed, sliding_speed=speed):
                        sample = Sample(seed, sliding_speed=speed, compliance=compliance)
                        self.assertGreater(np.linalg.norm(sample.force(np.zeros(3))), 1.0e-3)
                        matrix, shear = sample.anchor()
                        measured = sample.measured()
                        scale = np.abs(measured).max()
                        with_shear = np.abs(measured + (np.eye(3) + shear) @ matrix).max() / scale
                        without = np.abs(measured + matrix).max() / scale
                        # The symmetric tangent alone misses the change of the Coulomb bound.
                        self.assertLess(with_shear, 2.0e-3)
                        self.assertGreater(without, 5.0e-2)
                        # S = friction * sign * t d.T: rank one, of size friction, along the shaft,
                        # with the sign of the sliding.
                        self.assertAlmostEqual(np.linalg.norm(shear), FRICTION, places=5)
                        self.assertEqual(np.linalg.matrix_rank(shear, tol=1.0e-6), 1)
                        along = shear @ shear.T @ sample.tangent / FRICTION**2
                        np.testing.assert_allclose(along, sample.tangent, atol=1.0e-5)
                        opposite = Sample(seed, sliding_speed=-speed, compliance=compliance).anchor()
                        np.testing.assert_allclose(opposite[1], -shear, atol=1.0e-6)
                        np.testing.assert_array_equal(opposite[0], matrix)

    def test_sticking_unloaded_and_frictionless_samples_have_no_shear(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                sample = Sample(seed, stick=True)
                matrix, shear = sample.anchor()
                np.testing.assert_array_equal(shear, 0)
                measured = sample.measured()
                self.assertLess(np.abs(measured + matrix).max() / np.abs(measured).max(), 2.0e-3)
                # The shear scales with the friction coefficient of the correction and vanishes with it.
                sliding = Sample(seed, sliding_speed=0.005)
                np.testing.assert_allclose(sliding.anchor(0.1)[1], 0.5 * sliding.anchor(0.2)[1], atol=1.0e-7)
                np.testing.assert_array_equal(sliding.anchor(0.0)[1], 0)
                for unloaded in (dict(engaged=0), dict(length=0.0)):
                    matrix, shear = Sample(seed, sliding_speed=0.005, **unloaded).anchor()
                    np.testing.assert_array_equal(matrix, 0)
                    np.testing.assert_array_equal(shear, 0)


# ---------------------------------------------------------------------------------------------
# The package's operator, solver and correction over a small dense system.


def anchor_shears(matrices, friction=FRICTION):
    """A shear for every sliding tangent of the fixture (rank two: the transverse projection); zero otherwise.

    The fixture's sliding tangent is a * ((1 - f) (I - t t.T) + f d d.T): its null vector is the shaft
    tangent and the eigenvector of its largest eigenvalue the direction of the wall reaction.
    """
    shears = np.zeros_like(matrices)
    for index, matrix in enumerate(np.asarray(matrices, dtype=float)):
        if np.linalg.matrix_rank(matrix, tol=1.0e-5) != 2:
            continue
        _, vectors = np.linalg.eigh(matrix)
        shears[index] = friction * (-1.0) ** index * np.outer(vectors[:, 0], vectors[:, 2])
    return shears


class SyntheticShaft(_ShaftPreconditioner):
    """The shaft correction over the dense tissue block and the track of a ``Packet`` of the start-guess tests.

    Operator, diagonal, solver and balanced correction are the package's own; the anchors (tangents,
    levers, shears) are written by the test, since the fixture has no geometry for the anchor kernels.
    """

    def __init__(self, packet, shears, tolerance=1e-3):
        source = packet.pre
        device = source.device
        nodes = len(packet.fixed)
        self.packet = packet
        self.device, self.n, self.count = device, source.n, source.count
        # The anchors have one row for every sample of the packet: the launch size of the kernels over them.
        self.capacity = source.count
        self.fixed = wp.array(packet.fixed, dtype=int, device=device)
        # The tissue rows as blocks; the fixed rows of the fixture's matrix are rows of the identity.
        blocks = packet.mass[: 3 * nodes, : 3 * nodes].reshape(nodes, 3, nodes, 3).transpose(0, 2, 1, 3)
        rows, columns = np.divmod(np.arange(nodes * nodes), nodes)
        matrix = wp.sparse.bsr_from_triplets(
            nodes,
            nodes,
            wp.array(rows, dtype=int, device=device),
            wp.array(columns, dtype=int, device=device),
            wp.array(blocks.reshape(-1, 3, 3).astype(np.float32), dtype=wp.mat33, device=device),
        )
        self.coupler = SimpleNamespace(
            interaction=source.coupler.interaction, fem_solver=SimpleNamespace(_fixed_particles=None, A_bsr=matrix)
        )
        self.rhs, self.z = source.rhs, source.z
        self.work = wp.zeros_like(self.rhs)
        self.diagonal = wp.zeros(self.n, dtype=wp.mat33, device=device)
        self.inverse_diagonal = wp.zeros_like(self.diagonal)
        self.rigid_blocks = source.rigid_blocks
        self.time_step = source.time_step
        self._last_dt = H
        self.matrices, self.levers = source.matrices, source.levers
        self.shears = wp.array(shears, dtype=wp.mat33, device=device)
        # Condition of the device branch of a captured solve: the live track count of the interaction.
        live = source.coupler.interaction._live
        self._has_track = live[LIVE_TRACK : LIVE_TRACK + 1]
        self.operator = LinearOperator((self.n, self.n), wp.vec3, device, self._solver_matvec)
        self.inverse = LinearOperator((self.n, self.n), wp.vec3, device, self.precondition)
        self._conditional = bool(device.is_cuda and wp.is_conditional_graph_supported())
        self._solver_tolerance = tolerance
        self.nonsymmetric = bool(np.any(shears))
        self._symmetric_solver = self._make_linear_solver(cg)
        self._sheared_solver = self._make_linear_solver(bicgstab) if self.nonsymmetric else None
        self.linear_solver = self._anchor_solver()
        self._woodbury_guess = None
        self.residual = wp.zeros(3 * self.n, dtype=wp.float64, device=device)
        self.output = wp.zeros_like(self.residual)
        self.tangents, self.all_shears = packet.matrices.copy(), np.asarray(shears, dtype=np.float32).copy()
        self.set_tracks(self.count)

    def set_tracks(self, tracks):
        """Number of track samples, on the host and on the device; the anchors beyond it are zero.

        The kernels run the first ``tracks`` rows whatever the rows beyond them hold; the zeros
        make the whole arrays the anchors of the live system for the dense oracle.
        """
        interaction = self.coupler.interaction
        live = interaction._live.numpy()
        live[LIVE_TRACK] = tracks
        interaction._live.assign(live)
        self.count = tracks
        tangents, shears = self.tangents.copy(), self.all_shears.copy()
        tangents[tracks:], shears[tracks:] = 0, 0
        self.matrices.assign(tangents)
        self.shears.assign(shears)
        self.tracks = tracks

    def dense(self):
        """FP64 oracle of the live system: operator, full and free contact maps, (I + S) h D."""
        packet, h = self.packet, float(np.float32(H))
        full, _, _ = interpolation(packet.fixed, packet.nodes, packet.bary)
        free = full.copy()
        free[:, np.flatnonzero(np.repeat(packet.fixed, 3))] = 0
        rigid = rigid_interpolation(packet.levers)
        j_full, j_free = np.column_stack((-full, rigid)), np.column_stack((-free, rigid))
        tangents = self.matrices.numpy().astype(float)
        shears = self.shears.numpy().astype(float)
        action = h * block_diagonal([(np.eye(3) + s) @ d for s, d in zip(shears, tangents, strict=True)])
        return packet.mass + j_free.T @ action @ j_free, j_full, j_free, action

    def enqueue_diagonal(self):
        """The diagonal preparation of ``_enqueue_inputs`` after its anchor kernel."""
        interaction, matrix = self.coupler.interaction, self.coupler.fem_solver.A_bsr
        wp.launch(
            base_diagonal,
            self.n,
            [self.fixed, matrix.offsets, matrix.columns, matrix.values, self.rigid_blocks, self.diagonal],
            device=self.device,
        )
        wp.launch(
            contact_diagonal,
            self.capacity,
            [
                self.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                interaction._live,
                self.matrices,
                self.levers,
                self.shears,
                self.time_step,
                self.diagonal,
            ],
            device=self.device,
        )
        wp.launch(invert_diagonal, self.n, [self.diagonal, self.inverse_diagonal], device=self.device)

    def correction(self):
        """What a captured correction issues after its anchor kernel (``apply_preconditioner``)."""
        self.enqueue_diagonal()
        if self._woodbury_guess is not None:
            self._woodbury_guess.device_solve.enqueue()
        self._solve(self.residual, self.output)

    def load(self, rhs):
        self.rhs.assign(rhs.astype(np.float32))
        self.residual.assign(rhs.astype(np.float32).astype(np.float64).reshape(-1))

    def relative_residual(self):
        """|b - A z| / |b| of the device solution in the FP64 oracle."""
        operator = self.dense()[0]
        b = self.rhs.numpy().astype(float).reshape(-1)
        return np.linalg.norm(b - operator @ self.z.numpy().astype(float).reshape(-1)) / np.linalg.norm(b)

    def expected_output(self):
        """Balanced correction of the device solution: r - J_full.T (I + S) h D J_free z."""
        _, j_full, j_free, action = self.dense()
        z = self.z.numpy().astype(float).reshape(-1)
        return self.residual.numpy() - j_full.T @ action @ j_free @ z


def synthetic(seed=83, tracks=8, *, sheared=True, tolerance=1e-3):
    packet = woodbury.Packet(seed, tracks, ranks=(2, 3, 2, 0, 2))
    shears = anchor_shears(packet.matrices) if sheared else np.zeros_like(packet.matrices)
    return SyntheticShaft(packet, shears, tolerance)


class ShearedOperatorTests(unittest.TestCase):
    def test_operator_diagonal_and_correction_apply_the_sheared_tangent(self):
        pre = synthetic()
        self.assertEqual(int(np.any(pre.all_shears, axis=(1, 2)).sum()), 5)
        operator, j_full, j_free, action = pre.dense()
        scale = np.abs(operator).max()
        actual = dense_action(pre, "operator")
        np.testing.assert_allclose(actual, operator, rtol=0, atol=2.0e-6 * scale)
        # The shear makes the operator nonsymmetric; without it the same kernels give a symmetric one.
        self.assertGreater(np.abs(operator - operator.T).max(), 1.0e-3 * np.abs(action).max())
        self.assertGreater(np.abs(actual - actual.T).max(), 1.0e-3 * np.abs(action).max())
        pre.enqueue_diagonal()
        blocks = operator.reshape(pre.n, 3, pre.n, 3)[np.arange(pre.n), :, np.arange(pre.n), :]
        np.testing.assert_allclose(pre.diagonal.numpy(), blocks, rtol=0, atol=2.0e-6 * scale)
        rng = np.random.default_rng(5)
        pre.load(rng.normal(0, 1.0e-3, (pre.n, 3)))
        pre.z.assign(rng.normal(0, 0.1, (pre.n, 3)).astype(np.float32))
        output = wp.clone(pre.residual)
        interaction = pre.coupler.interaction
        wp.launch(
            balanced_correction,
            pre.capacity,
            [
                pre.fixed,
                interaction._track_nodes,
                interaction._track_bary,
                interaction._live,
                pre.matrices,
                pre.levers,
                pre.shears,
                pre.time_step,
                pre.z,
                output,
            ],
            device=pre.device,
        )
        expected = pre.expected_output()
        np.testing.assert_allclose(output.numpy(), expected, rtol=0, atol=2.0e-6 * np.abs(expected).max())
        plain = synthetic(sheared=False)
        self.assertIs(type(plain.linear_solver), CG)
        symmetric = dense_action(plain, "operator")
        self.assertLess(np.abs(symmetric - symmetric.T).max(), 2.0e-6 * scale)

    def test_nonsymmetric_solve_reaches_the_tolerance_of_the_symmetric_one(self):
        pre = synthetic()
        self.assertIs(type(pre.linear_solver), BiCGSTAB)
        rng = np.random.default_rng(11)
        for tracks in (8, 5):
            with self.subTest(tracks=tracks):
                pre.set_tracks(tracks)
                pre.load(rng.normal(0, 1.0e-3, (pre.n, 3)))
                pre.correction()
                # True residual against the tolerance of the solve (1e-3), with room for FP32.
                self.assertLess(pre.relative_residual(), 2.0e-3)
                expected = pre.expected_output()
                np.testing.assert_allclose(pre.output.numpy(), expected, rtol=0, atol=2.0e-6 * np.abs(expected).max())
                self.assertGreater(np.abs(pre.output.numpy() - pre.residual.numpy()).max(), 1.0e-6)


# ---------------------------------------------------------------------------------------------
# Solver selection by the anchor law.


def tunnel(interaction):
    return _TunnelALM(interaction, penalty_per_m=6000, compliance_per_m=1 / 6000, clearance=2e-5, tolerance=2e-6)


class SolverSelectionTests(unittest.TestCase):
    def assert_settings(self, pre, solver, kind, tolerance):
        """A stock solver object, with the settings the correction had before."""
        self.assertIs(type(solver), kind)
        conditional = bool(pre.device.is_cuda and wp.is_conditional_graph_supported())
        self.assertEqual(
            (solver._tol, solver._atol, solver._maxiter, solver._check_every, solver._use_cuda_graph),
            (tolerance, 0.0, 80, 0 if conditional else 10, conditional),
        )
        self.assertIs(solver._A, pre.operator)
        self.assertIs(solver._M, pre.inverse)
        self.assertIs(solver._b, pre.rhs)
        self.assertIs(solver._x, pre.z)

    def assert_solver(self, pre, kind, tolerance):
        """The solver of the anchor law; every correction also holds the CG of a symmetric operator."""
        self.assert_settings(pre, pre.linear_solver, kind, tolerance)
        self.assertEqual(pre.nonsymmetric, kind is BiCGSTAB)
        self.assertIs(pre.linear_solver, pre._sheared_solver if kind is BiCGSTAB else pre._symmetric_solver)
        self.assertIs(pre.linear_solver, pre._anchor_solver())
        self.assert_settings(pre, pre._symmetric_solver, CG, tolerance)
        if type(pre) is not SurfaceContactPreconditioner:
            # A correction that holds track anchors has the solver of sheared anchors from its
            # construction on, whatever law its track follows.
            self.assert_settings(pre, pre._sheared_solver, BiCGSTAB, tolerance)

    def test_shaft_correction_follows_the_anchor_law_not_the_friction_coefficient(self):
        with _example() as ex:
            h = 1 / 240
            setup(ex, h, 6000)
            interaction = ex.interaction
            self.assertIsNone(interaction.tunnel_alm)
            residual = wp.array(np.linspace(-2e-5, 3e-5, 3 * (ex.model.particle_count + 2)), dtype=wp.float64)
            # Penalty tunnel: the regularized friction tangent is symmetric.
            penalty = _ShaftPreconditioner(ex.coupler)
            self.assert_solver(penalty, CG, 1e-3)
            first = _apply(penalty, ex.state_0, h, residual).numpy()
            self.assertGreater(penalty.count, 0)
            self.assertGreater(np.abs(first - residual.numpy()).max(), 0)
            np.testing.assert_array_equal(penalty.shears.numpy(), 0)
            # No tunnel: no shaft law at all, the correction is the residual itself.
            stiffness, damping = interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m
            interaction.shaft_stiffness_per_m = interaction.shaft_damping_per_m = 0.0
            try:
                bare = _ShaftPreconditioner(ex.coupler)
                self.assert_solver(bare, CG, 1e-3)
                self.assertFalse(bare.active)
                self.assertEqual(_apply(bare, ex.state_0, h, residual).numpy().tobytes(), residual.numpy().tobytes())
            finally:
                interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m = stiffness, damping
            # Friction tunnel: BiCGSTAB by the presence of the model, also while its friction is zero.
            friction = interaction.friction
            interaction.tunnel_alm = tunnel(interaction)
            try:
                for coefficient in (0.0, friction):
                    interaction.friction = coefficient
                    sheared = _ShaftPreconditioner(ex.coupler)
                    self.assert_solver(sheared, BiCGSTAB, 1e-2)
                interaction.tunnel_alm.begin_step()
                output = _apply(sheared, ex.state_0, h, residual)
                self.assertTrue(np.isfinite(output.numpy()).all())
                self.assertGreater(np.abs(output.numpy() - residual.numpy()).max(), 0)
                # The anchors have the rows of the track reserve, not of the track samples.
                self.assertGreater(sheared.capacity, sheared.count)
                for anchors in (sheared.matrices, sheared.levers, sheared.shears):
                    self.assertEqual(anchors.shape, (interaction.track_capacity,))
                # Without track samples the operator of the same correction holds no anchor: CG.
                count, sheared.count = sheared.count, 0
                self.assert_settings(sheared, sheared._anchor_solver(), CG, 1e-2)
                sheared.count = count
                self.assertIs(sheared._anchor_solver(), sheared._sheared_solver)
                # A model attached after construction is solved by the BiCGSTAB that the correction
                # has held since its construction: the step that begins next makes no solver, loads
                # no kernel and records no capture. The CG and the anchor arrays stay the objects they
                # were, so nothing that an earlier graph addresses is released; the tolerance stays
                # the one of the construction.
                symmetric, count = penalty.linear_solver, penalty.count
                attached = penalty._sheared_solver
                self.assert_settings(penalty, attached, BiCGSTAB, 1e-3)
                anchors = (penalty.matrices, penalty.levers, penalty.shears)
                with (
                    patch.object(penalty, "_make_linear_solver", side_effect=AssertionError("solver made")),
                    patch.object(shaft_preconditioner, "ScopedCapture", side_effect=AssertionError("capture")),
                    patch.object(wp, "load_module", side_effect=AssertionError("module loaded")),
                ):
                    penalty.begin_step()
                self.assert_solver(penalty, BiCGSTAB, 1e-3)
                self.assertIs(penalty._symmetric_solver, symmetric)
                self.assertIs(penalty._sheared_solver, attached)
                self.assertIsNone(penalty._graph_key)
                self.assertEqual(penalty._graphs, {})
                self.assertTrue(np.isfinite(_apply(penalty, ex.state_0, h, residual).numpy()).all())
                penalty.begin_step()
                self.assertIs(penalty._sheared_solver, attached)
                self.assertEqual(penalty.count, count)
                for before, after in zip(anchors, (penalty.matrices, penalty.levers, penalty.shears), strict=True):
                    self.assertIs(after, before)
                # Shears as the tunnel anchors of sliding samples leave them.
                penalty.shears.fill_(wp.mat33(0.25))
            finally:
                interaction.tunnel_alm.discard()
                interaction.tunnel_alm = None
                interaction.friction = friction
            # And back: a removed model leaves no shear behind, in the same array, and the correction
            # of the penalty tunnel is the one it was before the model came.
            penalty.begin_step()
            self.assert_solver(penalty, CG, 1e-3)
            self.assertIs(penalty.linear_solver, symmetric)
            self.assertIs(penalty._sheared_solver, attached)
            self.assertIs(penalty.shears, anchors[2])
            np.testing.assert_array_equal(penalty.shears.numpy(), 0)
            again = _apply(penalty, ex.state_0, h, residual).numpy()
            np.testing.assert_array_equal(penalty.shears.numpy(), 0)
            if ex.device.is_cuda:
                # Sums over samples have no fixed order on CUDA.
                np.testing.assert_allclose(again, first, rtol=0, atol=1e-4 * np.abs(first - residual.numpy()).max())
            else:
                self.assertEqual(again.tobytes(), first.tobytes())

    def test_correction_of_a_track_without_samples_is_the_symmetric_solve(self):
        # The captured trial loop solves the system before the first puncture with the shaft correction
        # of a track that has no sample yet: no anchor, so no shear, and the solve is CG as it was.
        pre = synthetic()
        self.assertIs(pre.linear_solver, pre._sheared_solver)
        rng = np.random.default_rng(29)
        pre.load(rng.normal(0, 1.0e-3, (pre.n, 3)))
        tracks = pre.count
        pre.set_tracks(0)
        pre.correction()
        self.assertIs(pre.linear_solver, pre._symmetric_solver)
        self.assertIs(type(pre.linear_solver), CG)
        b, z = pre.rhs.numpy().astype(float).reshape(-1), pre.z.numpy().astype(float).reshape(-1)
        self.assertLess(np.linalg.norm(b - pre.packet.mass @ z), 2.0e-3 * np.linalg.norm(b))
        self.assertEqual(pre.output.numpy().tobytes(), pre.residual.numpy().tobytes())
        # The same object with its samples again: the sheared operator and its solver.
        pre.set_tracks(tracks)
        pre.correction()
        self.assertIs(pre.linear_solver, pre._sheared_solver)
        self.assertLess(pre.relative_residual(), 2.0e-3)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "Captured trial loop")
    def test_tunnel_attached_after_captured_steps_is_solved_by_new_trial_graphs(self):
        # No scene does this: its tunnel model exists before its coupler. A model attached later
        # changes the signature of the captured trial loop (the identity of the model is an entry),
        # so no graph of the other anchor law is launched again, and nothing such a graph
        # addresses is released: both solvers and the anchor arrays stay the objects they were.
        with _example() as example:
            coupled._configure(example, "penalty", True)
            coupler, interaction = example.coupler, example.interaction
            # The settings of the tunnel steps of these fixtures, from the first step on.
            interaction.friction, interaction.shaft_damping_per_m = 0.02, 0
            pre = coupler._shaft_preconditioner

            def step():
                coupler.step(example.state_0, example.state_1, example.control, coupled._DT)
                example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)

            def captured():
                trials = coupler._captured_trials
                graphs = [entry[0] for entry in trials.graphs.values()]
                return trials.signature, graphs, coupler.coupling_graph_captures

            def launched_again(graphs, earlier):
                return any(graph is other for graph in graphs for other in earlier)

            step()
            step()
            self.assert_solver(pre, CG, 1e-3)
            symmetric = pre.linear_solver
            signature, graphs, captures = captured()
            self.assertEqual((len(graphs), captures, coupler.coupling_graph_launches), (2, 2, 2))
            interaction.tunnel_alm = _TunnelALM(
                interaction, penalty_per_m=1500, compliance_per_m=1 / 1500, clearance=2e-5, tolerance=2e-6
            )
            interaction.tunnel_alm._resize()
            step()
            self.assert_solver(pre, BiCGSTAB, 1e-3)
            self.assertIs(pre._symmetric_solver, symmetric)
            sheared = pre._sheared_solver
            attached_signature, attached_graphs, attached_captures = captured()
            self.assertNotEqual(attached_signature, signature)
            self.assertEqual((attached_captures, coupler.coupling_graph_launches), (captures + 1, 3))
            self.assertFalse(launched_again(attached_graphs, graphs))
            step()
            attached_signature, attached_graphs, attached_captures = captured()
            self.assertEqual((len(attached_graphs), attached_captures), (2, captures + 2))
            anchors = (pre.matrices, pre.levers, pre.shears)
            # Shears as the tunnel anchors of sliding samples leave them.
            pre.shears.fill_(wp.mat33(0.25))
            interaction.tunnel_alm.discard()
            interaction.tunnel_alm = None
            step()
            self.assert_solver(pre, CG, 1e-3)
            self.assertIs(pre.linear_solver, symmetric)
            self.assertIs(pre._sheared_solver, sheared)
            removed_signature, removed_graphs, removed_captures = captured()
            self.assertNotEqual(removed_signature, attached_signature)
            self.assertEqual((removed_captures, coupler.coupling_graph_launches), (captures + 3, 5))
            self.assertFalse(launched_again(removed_graphs, graphs + attached_graphs))
            # The anchors have the rows of the track reserve: no step replaced one.
            for before, after in zip(anchors, (pre.matrices, pre.levers, pre.shears), strict=True):
                self.assertIs(after, before)
            np.testing.assert_array_equal(pre.shears.numpy(), 0)
            for name in ("particle_q", "body_q", "body_qd"):
                self.assertTrue(np.isfinite(getattr(example.state_0, name).numpy()).all(), name)

    def test_surface_only_metric_stays_symmetric_beside_a_friction_tunnel(self):
        with _example() as ex:
            _h, native, coupler = native_embedded(ex)
            self.assertIs(type(coupler._surface_preconditioner), SurfaceContactPreconditioner)
            self.assertIs(type(coupler._shaft_preconditioner), CombinedContactPreconditioner)
            self.assert_solver(coupler._surface_preconditioner, CG, 1e-3)
            self.assert_solver(coupler._shaft_preconditioner, CG, 1e-3)
            ex.interaction.tunnel_alm = tunnel(ex.interaction)
            try:
                sheared = NeedleTissueCoupler(
                    ex.model,
                    ex.solver,
                    native,
                    ex.interaction,
                    ex.needle_body,
                    ex.drive.com,
                    moment_length=ex.geometry.radius,
                    shaft_preconditioner=True,
                    surface_preconditioner=True,
                    preconditioned_anderson=True,
                    compact_rigid=False,
                )
                surface, combined = sheared._surface_preconditioner, sheared._shaft_preconditioner
                self.assertIs(type(surface), SurfaceContactPreconditioner)
                self.assertIs(type(combined), CombinedContactPreconditioner)
                # The facet metric holds no track anchor: its operator is symmetric and its solve CG.
                self.assert_solver(surface, CG, 1e-2)
                self.assertIsNone(surface._friction_tunnel())
                self.assertIsNone(surface.shears)
                self.assertIsNone(surface._sheared_solver)
                surface.begin_step()
                self.assert_solver(surface, CG, 1e-2)
                # The shaft correction of the same coupler holds the sheared anchors.
                self.assert_solver(combined, BiCGSTAB, 1e-2)
            finally:
                ex.interaction.tunnel_alm = None


# ---------------------------------------------------------------------------------------------
# Without a shear the changed kernels write the bits of their earlier text.

_PARAMETER = (r"    shears: wp\.array\[wp\.mat33\],\n", "", 1)
_EARLIER_TEXT = {
    "contact_action": [
        _PARAMETER,
        (r"    # A sliding tunnel anchor[^\n]*\n    force \+= shears\[i\] \* force\n", "", 1),
    ],
    "contact_diagonal": [_PARAMETER, (r"    matrix \+= shears\[i\] \* matrix\n", "", 1)],
    "balanced_correction": [_PARAMETER, (r"    force \+= shears\[i\] \* force\n", "", 1)],
    "prepare_alm_anchors": [
        _PARAMETER,
        (r"    shear = wp\.mat33\(0\.0\)\n", "", 1),
        (r"            else:\n(?:                #[^\n]*\n)+                shear = [^\n]*\n", "", 1),
        (r"    shears\[i\] = shear\n", "", 1),
    ],
}
_compiled = {}


def earlier_text(name):
    """Source of a changed kernel without the shear."""
    text = inspect.getsource(getattr(shaft_preconditioner, name).func)
    for pattern, replacement, places in _EARLIER_TEXT[name]:
        text, found = re.subn(pattern, replacement, text)
        if found != places:
            raise AssertionError(f"{name}: {pattern!r} matches {found} places, expected {places}")
    if "shear" in text:
        raise AssertionError(f"{name}: the earlier text still holds a shear")
    return text


def earlier_kernels():
    """Compile the earlier text of every changed kernel next to the globals of its module."""
    if not _compiled:
        namespace = dict(vars(shaft_preconditioner), __name__=f"{__name__}.shaft_preconditioner")
        for name in _EARLIER_TEXT:
            text = earlier_text(name)
            filename = f"<{namespace['__name__']}.{name}>"
            # Warp reads a kernel's source through inspect, which serves this text from the line cache.
            linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
            exec(compile(text, filename, "exec"), namespace)
            _compiled[name] = namespace[name]
    return _compiled


def named(kernel, inputs):
    parameters = list(inspect.signature(kernel.func).parameters)
    if len(parameters) != len(inputs):
        raise AssertionError(f"{kernel.key}: {len(inputs)} arguments for {len(parameters)} parameters")
    return dict(zip(parameters, inputs, strict=True))


def same_bits(name, kernel, dim, inputs, outputs):
    """Launch a kernel and its earlier text from the same start values; every array must hold the same bits."""
    values = named(kernel, inputs)
    if np.any(values["shears"].numpy()):
        raise AssertionError(f"{name}: compared with a shear")
    arrays = {key: value for key, value in values.items() if isinstance(value, wp.array) and value.size}
    start = {key: value.numpy().copy() for key, value in arrays.items()}
    wp.launch(kernel, dim, inputs, device=DEVICE)
    written = {key: value.numpy().copy() for key, value in arrays.items()}
    for key, value in arrays.items():
        value.assign(start[key])
    del values["shears"]
    wp.launch(earlier_kernels()[name], dim, list(values.values()), device=DEVICE)
    for key, value in arrays.items():
        if key != "shears" and value.numpy().tobytes() != written[key].tobytes():
            raise AssertionError(f"{name}: {key} differs from the earlier text")
        value.assign(written[key])
    changed = {key for key in arrays if written[key].tobytes() != start[key].tobytes()}
    if not set(outputs) <= changed:
        raise AssertionError(f"{name}: wrote {sorted(changed)}, expected {sorted(outputs)}")


@unittest.skipIf(wp.get_device(DEVICE).is_cuda, "bitwise on the CPU: sums over samples have no fixed order on CUDA")
class EarlierTextTests(unittest.TestCase):
    def test_kernels_without_a_shear_write_the_bits_of_their_earlier_text(self):
        earlier_kernels()
        # The three kernels that apply the anchors, on a track with sliding, sticking and unloaded samples.
        pre = synthetic(sheared=False)
        interaction = pre.coupler.interaction
        rng = np.random.default_rng(3)
        x = wp.array(rng.normal(0, 0.1, (pre.n, 3)).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        track = [pre.fixed, interaction._track_nodes, interaction._track_bary, interaction._live]
        track += [pre.matrices, pre.levers, pre.shears]
        out = wp.array(rng.normal(0, 0.1, (pre.n, 3)).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        same_bits("contact_action", contact_action, pre.count, [*track, pre.time_step, x, out], ["out"])
        diagonal = wp.array(rng.normal(0, 0.1, (pre.n, 3, 3)).astype(np.float32), dtype=wp.mat33, device=DEVICE)
        same_bits("contact_diagonal", contact_diagonal, pre.count, [*track, pre.time_step, diagonal], ["diagonal"])
        output = wp.array(rng.normal(0, 0.1, 3 * pre.n), dtype=wp.float64, device=DEVICE)
        same_bits("balanced_correction", balanced_correction, pre.count, [*track, pre.time_step, x, output], ["output"])
        # The tunnel anchors: the tangent and the lever are what they were on every branch.
        for seed in range(3):
            for options in (dict(sliding_speed=0.005), dict(sliding_speed=-0.005), dict(stick=True), dict(engaged=0)):
                for compliance in (1.0 / 300000.0, 0.0):
                    with self.subTest(seed=seed, compliance=compliance, **options):
                        sample = Sample(seed, compliance=compliance, **options)
                        matrices = wp.zeros(1, dtype=wp.mat33, device=DEVICE)
                        matrices.fill_(wp.mat33(-7.25))
                        levers = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
                        levers.fill_(wp.vec3(-7.25))
                        shears = wp.zeros(1, dtype=wp.mat33, device=DEVICE)
                        # Zero friction: the kernel's shear is zero on every branch, so the text
                        # without it must leave the same tangent. The tangent of a sliding sample
                        # does not depend on the coefficient (the sticking sample keeps 0.2 below).
                        friction = FRICTION if options.get("stick") else 0.0
                        same_bits(
                            "prepare_alm_anchors",
                            prepare_alm_anchors,
                            1,
                            [*sample.anchor_inputs(friction), matrices, levers, shears],
                            ["matrices", "levers"],
                        )

    def test_whole_correction_of_the_penalty_tunnel_holds_the_bits_of_the_earlier_kernels(self):
        earlier = earlier_kernels()
        real = wp.launch
        replaced = []

        def launch(kernel, dim, inputs, *args, **kwargs):
            if isinstance(kernel, wp.Kernel) and kernel.key in earlier and kernel.module is combine.module:
                values = named(kernel, inputs)
                del values["shears"]
                replaced.append(kernel.key)
                return real(earlier[kernel.key], dim, list(values.values()), *args, **kwargs)
            return real(kernel, dim, inputs, *args, **kwargs)

        with _example() as ex:
            h = 1 / 240
            setup(ex, h, 6000)
            pre = _ShaftPreconditioner(ex.coupler)
            self.assertIs(type(pre.linear_solver), CG)
            rng = np.random.default_rng(831)
            residual = wp.array(rng.normal(0, 1e-5, 3 * pre.n), dtype=wp.float64, device=ex.device)
            fields = ("matrices", "levers", "diagonal", "inverse_diagonal", "rhs", "z")

            def correction():
                output = _apply(pre, ex.state_0, h, residual)
                return {"output": output.numpy().copy(), **{name: getattr(pre, name).numpy().copy() for name in fields}}

            now = correction()
            with patch.object(wp, "launch", side_effect=launch):
                before = correction()
            # Operator in every Krylov iteration, the diagonal and the correction.
            self.assertGreater(replaced.count("contact_action"), 2)
            self.assertEqual((replaced.count("contact_diagonal"), replaced.count("balanced_correction")), (1, 1))
            self.assertGreater(np.abs(now["output"] - residual.numpy()).max(), 0)
            np.testing.assert_array_equal(pre.shears.numpy(), 0)
            for name, value in now.items():
                self.assertEqual(value.tobytes(), before[name].tobytes(), name)


# ---------------------------------------------------------------------------------------------
# A breakdown of the nonsymmetric solve.


def native_tunnel(example):
    """The embedded shaft with the guarded native rigid adapter and a friction tunnel made before the coupler."""
    h, native, _ = native_embedded(example)
    example.interaction.friction = 0.02
    example.interaction.tunnel_alm = tunnel(example.interaction)
    # Multipliers of the track that the accepted probes laid, as a first step would size them.
    example.interaction.tunnel_alm._resize()
    coupler = NeedleTissueCoupler(
        example.model,
        example.solver,
        native,
        example.interaction,
        example.needle_body,
        example.drive.com,
        moment_length=example.geometry.radius,
        shaft_preconditioner=True,
        surface_preconditioner=True,
        preconditioned_anderson=True,
        compact_rigid=False,
        max_iterations=160,
        absolute_tolerance=2e-7,
        relative_tolerance=1e-4,
    )
    return h, native, coupler


class BreakdownTests(unittest.TestCase):
    """BiCGSTAB divides by r0 . A M p. Where that vanishes the stock solver returns a non-finite solution.

    The test makes it vanish with an operator whose image is zero. The correction is handed on as it is,
    and the trial that applies it is rejected by the test for it that the trial loop already has: the native
    rigid adapter's guard on the host, the finiteness test of the candidate in the captured loop of the free
    needle, and the linear-residual certificate of the tissue solve in the captured loop of the articulated
    tools. The legacy free-needle path has no such test for the applied load: there the trials end at their
    cap, or at the step-resolution test of a needle that the load has thrown. Each way the step is rejected
    and the accepted state stands.
    """

    def run_broken(self, example, coupler, dt, error, message):
        pre = coupler._shaft_preconditioner
        self.assertIs(type(pre.linear_solver), BiCGSTAB)
        self.assertTrue(pre.active)
        applied = []

        def no_image(x, y, z, alpha, beta):
            # A x = 0: the first step length of the stock solver is rho / (r0 . A M p) = rho / 0.
            applied.append(alpha)
            wp.launch(combine, pre.n, [x, y, z, 0.0, beta], device=pre.device)

        interaction, alm = example.interaction, example.interaction.tunnel_alm
        fields = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
        state = {name: getattr(example.state_0, name).numpy().copy() for name in fields}
        native = _native_state(coupler.rigid_solver)
        duals = alm.committed_normal.numpy().copy(), alm.committed_tangent.numpy().copy()
        track = history(interaction)
        with (
            patch.object(pre, "matvec", side_effect=no_image),
            patch.object(interaction, "accept", wraps=interaction.accept) as accept,
            patch.object(alm, "commit", wraps=alm.commit) as commit,
        ):
            with self.assertRaisesRegex(error, message):
                coupler.step(example.state_0, example.state_1, example.control, dt)
        self.assertGreater(len(applied), 1)
        # The correction of the broken-down solve was handed on as it was: not finite.
        self.assertFalse(np.isfinite(pre.z.numpy()).all())
        self.assertFalse(np.isfinite(coupler._direction.numpy()).all())
        # The step is rejected as a whole: nothing accepted, no multiplier published, and the
        # accepted state, the rigid integration state and the material history are as before.
        self.assertEqual((accept.call_count, commit.call_count), (0, 0))
        self.assertFalse(alm._open)
        for name, values in state.items():
            np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), values, err_msg=name)
        np.testing.assert_array_equal(_native_state(coupler.rigid_solver), native)
        np.testing.assert_array_equal(alm.committed_normal.numpy(), duals[0])
        np.testing.assert_array_equal(alm.committed_tangent.numpy(), duals[1])
        self.assertEqual(history(interaction), track)

    def test_native_host_trial_with_the_broken_down_correction_is_rejected_by_the_rigid_guard(self):
        with _example() as example:
            h, _native, coupler = native_tunnel(example)
            self.run_broken(example, coupler, h, NeedleStepTooLarge, "Native QACC warning")
            # The first trial was evaluated; the trial that applied its correction was rejected.
            self.assertEqual(coupler.iterations, 1)
            # The same step from the same accepted state, with the operator back.
            coupler.step(example.state_0, example.state_1, example.control, h)
            self.assertLessEqual(coupler.iterations, coupler.max_iterations)
            self.assertTrue(np.isfinite(example.state_1.body_q.numpy()).all())

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "Captured trial loop")
    def test_captured_trial_with_the_broken_down_correction_is_rejected_as_not_finite(self):
        with _example() as example:
            coupled._configure(example, "alm", True)
            coupler = example.coupler
            self.run_broken(
                example, coupler, coupled._DT, NeedleCouplingNotConverged, "Nonfinite needle/tissue interface"
            )
            self.assertGreater(coupler.coupling_graph_launches, 0)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "Captured articulated trial loop")
    def test_articulated_captured_trial_with_the_broken_down_correction_is_rejected_by_the_tissue_certificate(self):
        # The captured loop of the articulated tools (the grasp scene). Its tissue solve carries a
        # linear-residual certificate. The trial that applies the non-finite correction fails it,
        # and that is the reason the rejected step gives: the host asks the certificate before it
        # names a non-finite state.
        import test_mjw_device_trials as articulated

        from newton_solver_suturebot._src.solvers.coupled.mujoco_warp_trials import CapturedArticulatedTrials

        def step(scene):
            scene.coupler.step(
                scene.state_0, scene.state_1, scene.control, articulated.DT, external_forces=scene._external_forces
            )

        with articulated._pair() as (_, scene):
            coupler, interaction = scene.coupler, scene.interaction
            pre, alm = coupler._shaft_preconditioner, interaction.tunnel_alm
            certificate = scene.solver._certified_linear
            self.assertTrue(pre.nonsymmetric and pre.active)
            self.assertGreater(interaction.track_count, 1)
            self.assertIs(type(pre._sheared_solver), BiCGSTAB)
            applied = []

            def no_image(x, y, z, alpha, beta):
                applied.append(alpha)
                wp.launch(combine, pre.n, [x, y, z, 0.0, beta], device=pre.device)

            # A carried load that the first trial does not balance, so that the trials need a correction.
            load = coupler._load.numpy().copy()
            load[-6] += 1e-4
            coupler._load.assign(load)
            saved = articulated._state(scene)
            duals = alm.committed_normal.numpy().copy(), alm.committed_tangent.numpy().copy()
            rejections = certificate.rejections
            with (
                patch.object(pre, "matvec", side_effect=no_image),
                patch.object(interaction, "accept", wraps=interaction.accept) as accept,
                patch.object(alm, "commit", wraps=alm.commit) as commit,
            ):
                with self.assertRaisesRegex(NeedleCouplingNotConverged, "FEM current-matrix residual failed"):
                    step(scene)
            trials = coupler._captured_trials
            self.assertIs(type(trials), CapturedArticulatedTrials)
            self.assertGreater(coupler.coupling_graph_launches, 0)
            self.assertGreater(len(applied), 1)
            # The correction of the broken-down solve was handed on as it was, and the trial that
            # applied it was refused by the certificate of its tissue solve.
            self.assertFalse(np.isfinite(pre.z.numpy()).all())
            self.assertFalse(np.isfinite(coupler._direction.numpy()).all())
            self.assertEqual(certificate.rejections, rejections + 1)
            self.assertEqual(int(trials.finite.numpy()[0]), 0)
            self.assertEqual((accept.call_count, commit.call_count), (0, 0))
            self.assertFalse(alm._open)
            # The step is rejected as a whole: accepted state, rigid integration state, carried
            # load, tissue start and material history are as before.
            current = articulated._state(scene)
            for group in ("input", "output", "integration"):
                for name, expected in saved[group].items():
                    np.testing.assert_array_equal(current[group][name], expected, err_msg=f"{group}.{name}")
            for name in ("metrics", "increment", "load"):
                np.testing.assert_array_equal(current[name], saved[name], err_msg=name)
            self.assertEqual(current["history"], saved["history"])
            np.testing.assert_array_equal(alm.committed_normal.numpy(), duals[0])
            np.testing.assert_array_equal(alm.committed_tangent.numpy(), duals[1])
            # The rejected attempt captured the operator without image. The same step from the same
            # accepted state on new trial graphs, with the operator back, is accepted.
            trials.invalidate()
            step(scene)
            self.assertTrue(np.isfinite(pre.z.numpy()).all())
            for name in articulated.FIELDS:
                self.assertTrue(np.isfinite(getattr(scene.state_1, name).numpy()).all(), name)

    def test_legacy_host_trials_with_the_broken_down_correction_end_in_a_rejected_step(self):
        # The host trial loop of the free-needle example does not test the load it applies. On the
        # CPU its plain native solver resets itself on the non-finite load, no trial converges and
        # the trials end at their cap; on CUDA its rigid integrator returns a needle that the
        # step-resolution test of the trial forces refuses. Either way the step is rejected.
        with _example() as example:
            coupled._configure(example, "alm", False)
            coupler = example.coupler
            coupler.max_iterations = 6
            self.run_broken(
                example,
                coupler,
                coupled._DT,
                (NeedleCouplingNotConverged, NeedleStepTooLarge),
                "did not converge|Nonfinite|Unresolved tip displacement",
            )
            self.assertEqual(coupler.coupling_graph_launches, 0)
            self.assertLessEqual(coupler.iterations, 6)
            coupler.max_iterations = 160
            coupler.step(example.state_0, example.state_1, example.control, coupled._DT)
            self.assertLessEqual(coupler.iterations, coupler.max_iterations)

    def test_the_solve_and_the_kernels_hold_no_finiteness_test(self):
        # The rejections above are the trial loops' own: neither the solve nor a changed kernel tests for it.
        sources = [inspect.getsource(getattr(shaft_preconditioner, name).func) for name in _EARLIER_TEXT]
        sources += [inspect.getsource(_ShaftPreconditioner._solve), inspect.getsource(_ShaftPreconditioner.matvec)]
        sources += [
            inspect.getsource(getattr(_ShaftPreconditioner, name))
            for name in ("_make_linear_solver", "_anchor_solver", "_krylov", "begin_step")
        ]
        for text in sources:
            self.assertIsNone(re.search(r"isfinite|isnan|isinf|clamp|nan_to_num", text), text)


# ---------------------------------------------------------------------------------------------
# One captured correction at changing numbers of track samples; kernels loaded with the solver.


@unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA graph replay contract")
class CapturedCorrectionTests(unittest.TestCase):
    def guessed(self):
        """The synthetic correction with the device start guess, as a captured trial loop binds it.

        The start guess solves the sheared system on the track support. The load of these tests is on
        every node, so the guess leaves the part of it outside that support, 0.53 of the right-hand
        side here (measured on the CPU), and the nonsymmetric solve iterates to its tolerance of 1e-4.
        """
        pre = synthetic(seed=61, tolerance=1e-4)
        helper = WoodburyInitialGuess(pre)
        key = helper.prepare_device(pre.packet.manager)
        self.assertIsNotNone(key)
        # Five to eight samples share one factorization, so one graph serves them.
        self.assertEqual(helper.device_solve.capacity, 8)
        pre._woodbury_guess = helper
        return pre, helper, key

    def test_one_captured_correction_follows_the_device_track_count(self):
        pre, helper, key = self.guessed()
        rng = np.random.default_rng(17)
        pre.load(rng.normal(0, 1.0e-3, (pre.n, 3)))

        def correction():
            helper.device_solve.enqueue_topology()
            pre.correction()

        def reset():
            pre.z.fill_(wp.vec3(7.0))
            pre.output.fill_(-7.25)
            helper.coefficients.fill_(np.nan)
            helper.selected.fill_(-1)

        # Captured with the fewest samples: a launch sized by that count would miss the later ones.
        pre.set_tracks(5)
        with ScopedCapture(device=pre.device) as capture:
            correction()
        seen = []
        for tracks in (5, 8, 6, 7, 8):
            with self.subTest(tracks=tracks):
                pre.set_tracks(tracks)
                counters = helper.counters.numpy()
                reset()
                wp.capture_launch(capture.graph)
                replayed = {"z": pre.z.numpy().copy(), "output": pre.output.numpy().copy()}
                # The start guess of the sheared system on the track support passes its certificate;
                # the load is on every node, so the nonsymmetric solve starts from it, iterates and
                # ends below its cap.
                self.assertEqual(helper.selected.numpy()[0], 1)
                change = helper.counters.numpy() - counters
                self.assertEqual(change[:2].tolist(), [1, 1])
                self.assertGreaterEqual(change[2], 1)
                self.assertEqual(change[3], 0)
                self.assertLess(pre.relative_residual(), 2.0e-4)
                expected = pre.expected_output()
                np.testing.assert_allclose(replayed["output"], expected, rtol=0, atol=2.0e-6 * np.abs(expected).max())
                # The same launches issued eagerly at this count solve the same system to the same tolerance.
                reset()
                correction()
                self.assertLess(pre.relative_residual(), 2.0e-4)
                operator = pre.dense()[0]
                difference = (pre.z.numpy().astype(float) - replayed["z"]).reshape(-1)
                self.assertLess(np.linalg.norm(operator @ difference), 4.0e-4 * np.linalg.norm(pre.rhs.numpy()))
                seen.append(replayed["output"])
        # The replays differ with the count (by 0.46 of the correction on the CPU) and repeat with it.
        scale = np.abs(seen[0] - pre.residual.numpy()).max()
        for first, second in ((0, 1), (1, 2), (2, 3)):
            self.assertGreater(np.abs(seen[first] - seen[second]).max(), 1.0e-1 * scale)
        np.testing.assert_allclose(seen[4], seen[1], rtol=0, atol=1.0e-2 * scale)
        self.assertEqual(helper.prepare_device(pre.packet.manager), key)

    def test_one_captured_solve_takes_its_solver_by_the_live_track_count_of_each_launch(self):
        # The solve of a friction tunnel, captured at an empty track, as the trial loop of a scene
        # captures it before the first puncture. The number of track samples is device data, so the
        # same graph solves the sheared system by BiCGSTAB once the track has samples, and the
        # system without anchors by CG while it has none.
        pre = synthetic()
        rng = np.random.default_rng(41)
        pre.load(rng.normal(0, 1.0e-3, (pre.n, 3)))
        solvers = (pre._symmetric_solver, pre._sheared_solver)
        self.assertEqual([type(solver) for solver in solvers], [CG, BiCGSTAB])
        pre.set_tracks(0)
        with ScopedCapture(device=pre.device) as capture:
            pre.correction()
        # The solver that the host names is the one of its count; the graph holds both.
        self.assertIs(pre.linear_solver, pre._symmetric_solver)
        sentinel = -7
        for tracks in (0, 8, 5, 0, 8):
            with self.subTest(tracks=tracks):
                pre.set_tracks(tracks)
                for solver in solvers:
                    solver._cur_iter_and_condition.fill_(sentinel)
                pre.z.fill_(wp.vec3(7.0))
                pre.output.fill_(-7.25)
                wp.capture_launch(capture.graph)
                ran = [int(solver._cur_iter_and_condition.numpy()[0]) != sentinel for solver in solvers]
                self.assertEqual(ran, [tracks == 0, tracks > 0])
                if tracks:
                    self.assertGreaterEqual(int(pre._sheared_solver._cur_iter_and_condition.numpy()[0]), 1)
                    # True residual against the tolerance of the solve (1e-3), with room for FP32.
                    self.assertLess(pre.relative_residual(), 2.0e-3)
                    expected = pre.expected_output()
                    np.testing.assert_allclose(
                        pre.output.numpy(), expected, rtol=0, atol=2.0e-6 * np.abs(expected).max()
                    )
                    self.assertGreater(np.abs(pre.output.numpy() - pre.residual.numpy()).max(), 1.0e-6)
                else:
                    # No anchor: the tissue and rigid blocks alone, and the residual is the correction.
                    b, z = pre.rhs.numpy().astype(float).reshape(-1), pre.z.numpy().astype(float).reshape(-1)
                    self.assertLess(np.linalg.norm(b - pre.packet.mass @ z), 2.0e-3 * np.linalg.norm(b))
                    self.assertEqual(pre.output.numpy().tobytes(), pre.residual.numpy().tobytes())

    def test_solver_kernels_are_created_and_loaded_with_the_solver(self):
        pre, helper, _key = self.guessed()
        self.assertIs(type(pre.linear_solver), BiCGSTAB)
        self.assertIs(type(pre._symmetric_solver), CG)

        # Every module with a kernel of Warp's Krylov file: the module of the solver kernels, and the
        # modules of their dot products and of the loop condition, which Warp keeps one kernel each.
        # A launch loads the module of its kernel by itself (it does not go through wp.load_module),
        # so the loaded executables are compared: a kernel created by a correction gives the module of
        # the solver kernels a new hash, and with it a new executable at the next launch.
        def krylov_modules():
            return [
                module
                for module in list(sys.modules["warp._src.context"].user_modules.values())
                if any(
                    getattr(kernel.func, "__module__", None) == BiCGSTAB.__module__
                    for kernel in module.kernels.values()
                )
            ]

        krylov = krylov_modules()
        used = [module for module in krylov if any(context is pre.device.context for context, _ in module.execs)]
        names = " ".join(sorted(kernel.key for module in used for kernel in module.kernels.values()))
        for kernel in (
            "_bicgstab_kernel_1",
            "_cg_kernel_1",
            "block_dot_kernel",
            "block_sum_kernel",
            "_update_condition",
        ):
            self.assertIn(kernel, names)
        modules = [*krylov, combine.module]
        self.assertTrue(any(context is pre.device.context for context, _ in combine.module.execs))
        loaded = [dict(module.execs) for module in modules]
        rng = np.random.default_rng(23)
        for tracks in (pre.count, 0):
            # First correction of each solver of this object, issued as a trial loop captures it:
            # BiCGSTAB with the track samples, CG for a track without samples.
            pre.set_tracks(tracks)
            pre.load(rng.normal(0, 1.0e-3, (pre.n, 3)))
            with ScopedCapture(device=pre.device, force_module_load=False) as capture:
                if tracks:
                    helper.device_solve.enqueue_topology()
                    pre.correction()
                else:
                    pre.enqueue_diagonal()
                    pre._woodbury_guess = None
                    pre._solve(pre.residual, pre.output)
            wp.capture_launch(capture.graph)
            self.assertIs(pre.linear_solver, pre._sheared_solver if tracks else pre._symmetric_solver)
            if tracks:
                self.assertGreaterEqual(helper.counters.numpy()[2], 1)
                self.assertLess(pre.relative_residual(), 2.0e-4)
            else:
                b, z = pre.rhs.numpy().astype(float).reshape(-1), pre.z.numpy().astype(float).reshape(-1)
                self.assertLess(np.linalg.norm(b - pre.packet.mass @ z), 2.0e-4 * np.linalg.norm(b))
            # No kernel was created for the solve and no module loaded: the executables are the
            # ones that were loaded when the solvers were made.
            self.assertEqual([module.name for module in krylov_modules()], [module.name for module in krylov])
            for module, execs in zip(modules, loaded, strict=True):
                self.assertEqual(set(module.execs), set(execs), module.name)
                for variant, executable in execs.items():
                    self.assertIs(module.execs[variant], executable, f"{module.name} {variant}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
