# SPDX-License-Identifier: Apache-2.0
"""Changed FEM timesteps agree with fresh full matrix assembly."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot import FEMBuilder, SolverFEM


@wp.kernel
def _sum_x(q: wp.array[wp.vec3], total: wp.array[wp.float64]):
    wp.atomic_add(total, 0, wp.float64(q[wp.tid()][0]))


def dense(matrix):
    offsets, columns, blocks = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for index in range(offsets[row], offsets[row + 1]):
            column = columns[index]
            result[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += blocks[index]
    return result


class TestSolverTimestep(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    dt = 0.002

    def model(self, *, mixed=True):
        builder = FEMBuilder(gravity=(0, 0, 0))
        corners = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
        for point in 0.1 * corners:
            builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0.0), mass=0.001)
        builder.add_tetrahedron(0, 1, 3, 4, k_mu=500, k_lambda=1000, k_damp=0)
        if mixed:
            builder.add_soft_hex_mesh(np.arange(8).reshape(1, 8), k_mu=500, k_lambda=1000, k_damp=0)
            builder.add_triangle(0, 1, 3, tri_ke=30, tri_ka=50, tri_kd=0)
            builder.add_spring(2, 6, ke=40, kd=0.2, control=0)
        return builder.finalize(device=self.device)

    def solver(self, model, dt=None, *, differentiable=False, fixed=True):
        return SolverFEM(
            model,
            dt=self.dt if dt is None else dt,
            solver_type="cg",
            preconditioner_type="block_diag",
            linear_solver_maxiter=80,
            linear_solver_tolerance=1e-5,
            differentiable=differentiable,
            fixed_particle_indices=(0, 1) if fixed else None,
        )

    def test_full_matrix_and_physical_update_match_fresh_solvers(self):
        # Exercise both an existing sparse-slot map and a dt change on the very
        # first refresh, when construction has pruned numerically zero blocks.
        for prepare_initial in (False, True):
            with self.subTest(prepare_initial=prepare_initial):
                model = self.model()
                actual = self.solver(model)
                current, following = model.state(), model.state()
                q = model.particle_q.numpy().copy()
                q[2] += (0.002, -0.001, 0.001)
                q[6] += (-0.001, 0.002, -0.003)
                velocity = np.linspace(-0.02, 0.03, 24, dtype=np.float32).reshape(8, 3)
                velocity[:2] = 0
                force = np.linspace(-0.01, 0.02, 24, dtype=np.float32).reshape(8, 3)
                external = wp.array(force, dtype=wp.vec3, device=self.device)
                current.particle_q.assign(q)
                if prepare_initial:
                    actual.refresh_tangent(current.particle_q)
                storage = None
                timesteps = (self.dt, self.dt / 2, self.dt) if prepare_initial else (self.dt / 2, self.dt)
                for dt in timesteps:
                    reference = self.solver(model, dt)
                    reference_in, reference_out = model.state(), model.state()
                    for state, solver in ((current, actual), (reference_in, reference)):
                        state.particle_q.assign(q)
                        state.particle_qd.assign(velocity)
                        solver.velocity_increment.assign(0.25 * velocity)
                        solver.refresh_tangent(state.particle_q, dt)
                    self.assertEqual(actual.dt, dt)
                    np.testing.assert_allclose(
                        actual.bsr_values.numpy(), reference.bsr_values.numpy(), rtol=1e-6, atol=1e-9
                    )
                    np.testing.assert_allclose(dense(actual.A_bsr), dense(reference.A_bsr), rtol=1e-6, atol=1e-9)
                    pointers = (
                        actual.A_bsr.values.ptr,
                        actual.A_bsr.columns.ptr,
                        actual.A_bsr.offsets.ptr,
                        actual._pc_inv_diag.ptr,
                    )
                    if storage is not None:
                        self.assertEqual(pointers, storage)
                    storage = pointers
                    # Timestep changes now clear the old solver's Krylov
                    # guess; compare both assemblies with that same guess.
                    reference.velocity_increment.assign(actual.velocity_increment)
                    actual.step(current, following, None, None, dt, external_forces=external)
                    reference.step(reference_in, reference_out, None, None, dt, external_forces=external)
                    np.testing.assert_allclose(
                        actual.velocity_increment.numpy(), reference.velocity_increment.numpy(), rtol=2e-5, atol=1e-7
                    )
                    np.testing.assert_allclose(
                        following.particle_qd.numpy(), reference_out.particle_qd.numpy(), rtol=2e-5, atol=1e-7
                    )
                    np.testing.assert_allclose(
                        following.particle_q.numpy(), reference_out.particle_q.numpy(), rtol=1e-6, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        current.particle_f.numpy(), reference_in.particle_f.numpy(), rtol=1e-6, atol=1e-9
                    )
                    np.testing.assert_array_equal(following.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
                    np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
                    # Omitted dt now retains the most recently assembled step.
                    actual.refresh_tangent(current.particle_q)
                    self.assertEqual(actual.dt, dt)

    def test_differentiable_and_tape_timestep_changes_are_rejected(self):
        model = self.model()
        solver = self.solver(model, differentiable=True, fixed=False)
        before = dense(solver.A_bsr)
        with self.assertRaisesRegex(ValueError, "Differentiable"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual(solver.dt, self.dt)
        np.testing.assert_array_equal(dense(solver.A_bsr), before)
        solver = self.solver(model)
        before = dense(solver.A_bsr)
        with wp.Tape(), self.assertRaisesRegex(RuntimeError, "outside tapes"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual(solver.dt, self.dt)
        np.testing.assert_array_equal(dense(solver.A_bsr), before)

    def test_rejected_timestep_is_not_written_and_leaves_taped_gradients_unchanged(self):
        # The launches of a step read the timestep from solver.step_time, and the taped
        # integration kernel reads it again in tape.backward(). A request the solver rejects
        # must therefore never reach that array: one free particle, x_out = x + dt (v + dv),
        # so d(sum x_out)/d(v_in) is the timestep the step was recorded with.
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_particle((0, 0, 1), (0.5, 0, 0), mass=2.0)
        model = builder.finalize(device=self.device)
        solver = self.solver(model, differentiable=True, fixed=False)
        initial, output = model.state(requires_grad=True), model.state(requires_grad=True)
        scratch = model.state(), model.state()
        loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        with wp.Tape() as tape:
            solver.step(initial, output, None, None, self.dt)
            wp.launch(_sum_x, 1, inputs=[output.particle_q], outputs=[loss], device=self.device)
        held = solver.step_time.seconds.numpy().copy()
        with self.assertRaisesRegex(ValueError, "Differentiable dt must match"):
            solver.step(*scratch, None, None, self.dt / 2)
        with self.assertRaisesRegex(ValueError, "Differentiable"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual((solver.step_time.value, solver.step_time.writes), (self.dt, 1))
        np.testing.assert_array_equal(solver.step_time.seconds.numpy(), held)
        tape.backward(loss)
        np.testing.assert_array_equal(initial.particle_qd.grad.numpy()[0], np.float32([self.dt, 0, 0]))

        # The same for the other requests that are refused: nothing is written before the refusal.
        model = self.model()
        solver = self.solver(model)
        current, following = model.state(), model.state()
        with wp.Tape(), self.assertRaisesRegex(RuntimeError, "differentiable=True"):
            solver.step(current, following, None, None, self.dt / 2)
        wrong = wp.zeros(model.particle_count + 1, dtype=wp.vec3, device=self.device)
        with self.assertRaisesRegex(ValueError, "particle shape"):
            solver.refresh_tangent(wrong, self.dt / 2)
        with self.assertRaisesRegex(ValueError, "external_forces"):
            solver.step(current, following, None, None, self.dt / 2, external_forces=wrong)
        self.assertEqual((solver.dt, solver.step_time.value, solver.step_time.writes), (self.dt, self.dt, 1))
        np.testing.assert_array_equal(solver.step_time.seconds.numpy(), np.float32([self.dt]))

    def test_captured_refresh_and_step_follow_a_timestep_changed_outside_capture(self):
        # A caller's graph of a tangent refresh and a step, as the phantom examples capture.
        # The element blocks are assembled inside the graph from the timestep in
        # solver.step_time; refresh_tangent(q, dt) outside capture rebuilds the blocks that
        # no graph holds and writes that timestep. The one graph then gives the step of a
        # fresh solver at every timestep. With the timestep of the capture kept in the
        # element kernels, the replays at another one would assemble dt0² K.
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA graph capture")
        model = self.model()
        actual = self.solver(model)
        current, following = model.state(), model.state()
        q = model.particle_q.numpy().copy()
        q[2] += (0.002, -0.001, 0.001)
        q[6] += (-0.001, 0.002, -0.003)
        velocity = np.linspace(-0.02, 0.03, 24, dtype=np.float32).reshape(8, 3)
        velocity[:2] = 0
        external = wp.array(
            np.linspace(-0.01, 0.02, 24, dtype=np.float32).reshape(8, 3), dtype=wp.vec3, device=self.device
        )

        def frame():
            actual.refresh_tangent(current.particle_q)
            actual.step(current, following, None, None, actual.dt, external_forces=external)

        def start(state, solver):
            state.particle_q.assign(q)
            state.particle_qd.assign(velocity)
            solver.velocity_increment.zero_()

        # Sparse structure, Krylov storage and kernels first; then one capture.
        start(current, actual)
        frame()
        start(current, actual)
        with wp.ScopedCapture(device=self.device) as capture:
            frame()
            # What still takes the timestep as a launch argument, the mass, spring and triangle
            # blocks, cannot enter a graph; nor can a write of another timestep.
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                actual._build_constant_matrix(model, actual.dt)
            with self.assertRaisesRegex(RuntimeError, "outside tapes and CUDA graph capture"):
                actual.refresh_tangent(current.particle_q, self.dt / 2)
            with self.assertRaisesRegex(RuntimeError, "during a CUDA graph capture"):
                actual.step(current, following, None, None, self.dt / 2, external_forces=external)
        self.assertEqual((actual.dt, actual.step_time.value), (self.dt, self.dt))
        pointers = (actual.A_bsr.values.ptr, actual._pc_inv_diag.ptr, actual.step_time.seconds.ptr)
        results = {}
        for dt in (self.dt, self.dt / 2, self.dt / 3, 0.0020833333333333355 / 2, self.dt):
            with self.subTest(dt=dt):
                reference = self.solver(model, dt)
                reference_in, reference_out = model.state(), model.state()
                start(reference_in, reference)
                reference.refresh_tangent(reference_in.particle_q, dt)
                reference.step(reference_in, reference_out, None, None, dt, external_forces=external)
                # Host preparation of the replay: the constant blocks and the step time.
                actual.refresh_tangent(current.particle_q, dt)
                self.assertEqual((actual.dt, actual.step_time.value), (dt, dt))
                start(current, actual)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(dense(actual.A_bsr), dense(reference.A_bsr), rtol=1e-6, atol=1e-9)
                np.testing.assert_allclose(
                    actual.velocity_increment.numpy(), reference.velocity_increment.numpy(), rtol=2e-5, atol=1e-7
                )
                np.testing.assert_allclose(
                    following.particle_qd.numpy(), reference_out.particle_qd.numpy(), rtol=2e-5, atol=1e-7
                )
                np.testing.assert_allclose(
                    following.particle_q.numpy(), reference_out.particle_q.numpy(), rtol=1e-6, atol=1e-8
                )
                self.assertEqual(
                    pointers, (actual.A_bsr.values.ptr, actual._pc_inv_diag.ptr, actual.step_time.seconds.ptr)
                )
                results.setdefault(dt, (dense(actual.A_bsr), following.particle_qd.numpy().copy()))
        # The timesteps did differ in their effect, so a stale one would have been seen above.
        matrix, speed = results[self.dt]
        halved_matrix, halved_speed = results[self.dt / 2]
        self.assertFalse(np.allclose(halved_matrix, matrix, rtol=1e-2, atol=0))
        self.assertFalse(np.allclose(halved_speed, speed, rtol=1e-2, atol=1e-6))

    def test_changed_timestep_discards_only_the_numerical_warm_start(self):
        model = self.model()
        solver = self.solver(model)
        state = model.state()
        initial_q, initial_qd = state.particle_q.numpy(), state.particle_qd.numpy()
        guess = np.full((model.particle_count, 3), 0.02, dtype=np.float32)
        solver.velocity_increment.assign(guess)
        solver.refresh_tangent(state.particle_q)
        np.testing.assert_array_equal(solver.velocity_increment.numpy(), guess)
        solver.refresh_tangent(state.particle_q, self.dt / 2)
        np.testing.assert_array_equal(solver.velocity_increment.numpy(), np.zeros_like(guess))
        np.testing.assert_array_equal(state.particle_q.numpy(), initial_q)
        np.testing.assert_array_equal(state.particle_qd.numpy(), initial_qd)

    def test_streaming_and_invalid_timestep_changes_are_rejected(self):
        model = self.model(mixed=False)
        solver = self.solver(model, fixed=False)
        solver.prepare_tetrahedral_updates(model.particle_count)
        with self.assertRaisesRegex(ValueError, "Streaming"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual(solver.dt, self.dt)
        solver = self.solver(model)
        before = dense(solver.A_bsr)
        for dt in (0, -1, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "finite and positive"):
                solver.refresh_tangent(model.particle_q, dt)
            self.assertEqual(solver.dt, self.dt)
            np.testing.assert_array_equal(dense(solver.A_bsr), before)


if __name__ == "__main__":
    unittest.main()
