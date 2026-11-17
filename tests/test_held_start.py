"""Original FEM equations and compact solve stay authoritative after caching."""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp
from test_fem_prepared_step import TestPreparedFEMStep

from newton_solver_suturebot._src.solvers.fem.held_start import _GRAPH_LIMIT, HeldFEMStart
from newton_solver_suturebot._src.solvers.fem.solver import SolverFEM
from newton_solver_suturebot._src.utils.capture import ScopedCapture


class HeldStartTests(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def fixture(self, *, mixed=True):
        source = TestPreparedFEMStep()
        source.device = self.device
        model, state, make = source.fixture(mixed=mixed)
        actual, reference = make(), make()
        actual.refresh_tangent(state.particle_q)
        reference.refresh_tangent(state.particle_q)
        incoming = model.state()
        incoming.assign(state)
        control = model.control()
        cache = HeldFEMStart(actual)
        actual._held_start = cache
        return source.dt, model, state, incoming, control, actual, reference, cache

    def test_repeated_mixed_rhs_and_endpoint_parity_without_reassembly(self):
        dt, model, state, incoming, control, actual, reference, cache = self.fixture()
        before = [a.numpy().copy() for a in (state.particle_q, state.particle_qd, state.particle_f)]
        self.assertTrue(cache.prepare(state, control, None, dt))
        for array, value in zip((state.particle_q, state.particle_qd, state.particle_f), before, strict=True):
            np.testing.assert_array_equal(array.numpy(), value)
        cached_impulse = cache.state.particle_f.numpy().copy()
        for scale in (0.0, 1.0, -0.5):
            load = wp.array(scale * np.linspace(-0.01, 0.02, 24).reshape(8, 3), dtype=wp.vec3, device=self.device)
            actual.velocity_increment.zero_()
            reference.velocity_increment.zero_()
            out, want = model.state(), model.state()
            actual.step(state, out, control, None, dt, external_forces=load)
            reference.step(incoming, want, control, None, dt, external_forces=load)
            np.testing.assert_array_equal(state.particle_f.numpy(), incoming.particle_f.numpy())
            np.testing.assert_array_equal(out.particle_q.numpy(), want.particle_q.numpy())
            np.testing.assert_array_equal(out.particle_qd.numpy(), want.particle_qd.numpy())
            self.assertEqual(actual._lin_solver.compact_solve, reference._lin_solver.compact_solve)
            np.testing.assert_array_equal(cache.state.particle_f.numpy(), cached_impulse)
        self.assertEqual(cache.preparations, 1)
        self.assertEqual(cache.diagnostics()["cached_trial_assemblies"], 3)

    def test_new_state_and_tangent_are_reprepared_and_fixed_supports_preserved(self):
        dt, model, state, incoming, control, actual, reference, cache = self.fixture(mixed=False)
        cache.prepare(state, control, None, dt)
        pointers = tuple(a.ptr for a in cache.owners)
        q = state.particle_q.numpy()
        q[4] += (0.002, -0.001, 0.003)
        v = state.particle_qd.numpy()
        v[5] += (0.01, 0.02, -0.01)
        state.particle_q.assign(q)
        state.particle_qd.assign(v)
        incoming.assign(state)
        dt *= 0.5
        actual.refresh_tangent(state.particle_q, dt)
        reference.refresh_tangent(state.particle_q, dt)
        self.assertFalse(cache.matches(state, dt, control))
        self.assertIsNone(cache.capture_key())
        cache.prepare(state, control, None, dt)
        self.assertEqual(pointers, tuple(a.ptr for a in cache.owners))
        actual.velocity_increment.zero_()
        reference.velocity_increment.zero_()
        out, want = model.state(), model.state()
        actual.step(state, out, control, None, dt)
        reference.step(incoming, want, control, None, dt)
        np.testing.assert_array_equal(out.particle_q.numpy(), want.particle_q.numpy())
        np.testing.assert_array_equal(out.particle_qd.numpy(), want.particle_qd.numpy())
        np.testing.assert_array_equal(out.particle_q.numpy()[:2], model.particle_q.numpy()[:2])

    def test_end_attempt_custom_force_and_tape_fall_back(self):
        dt, model, state, incoming, control, actual, reference, cache = self.fixture(mixed=False)
        cache.prepare(state, control, None, dt)
        cache.invalidate()
        self.assertFalse(cache.matches(state, dt, control))
        self.assertIsNone(cache.capture_key())
        with mock.patch.object(actual, "_accumulate_forces", wraps=actual._accumulate_forces):
            self.assertFalse(cache.prepare(state, control, None, dt))
        with wp.Tape():
            self.assertFalse(cache.prepare(state, control, None, dt))
        self.assertFalse(cache.active)
        out, want = model.state(), model.state()
        actual.step(state, out, control, None, dt)
        reference.step(incoming, want, control, None, dt)
        np.testing.assert_array_equal(out.particle_qd.numpy(), want.particle_qd.numpy())

    def test_capture_replays_fresh_preparation_through_original_bicgstab_solver(self):
        self._capture_replays("bicgstab")

    def test_capture_replays_fresh_preparation_through_original_compact_cg_solver(self):
        self._capture_replays("cg")

    def _capture_replays(self, solver_kind):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA capture")
        dt, model, state, incoming, control, actual, reference, cache = self.fixture(mixed=False)
        for solver in (actual, reference):
            solver.solver_type = solver_kind
            solver.linear_solver_check_every = 0
            solver.linear_solver_use_cuda_graph = True
            solver.linear_solver_compact = True
            solver.linear_solver_chebyshev_degree = 4
        cache.prepare(state, control, None, dt)
        external = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        out, want = model.state(), model.state()
        actual.step(state, out, control, None, dt, external_forces=external)
        self.assertEqual(actual._lin_solver.compact_solve, solver_kind == "cg")
        with ScopedCapture(device=self.device) as captured:
            actual.step(state, out, control, None, dt, external_forces=external)
        for scale in (1.0, -2.0):
            velocity = state.particle_qd.numpy()
            velocity[5] += np.float32(scale * 0.003)
            state.particle_qd.assign(velocity)
            incoming.assign(state)
            cache.prepare(state, control, None, dt)
            external.fill_(wp.vec3(0.001 * scale, -0.002 * scale, 0.003 * scale))
            actual.velocity_increment.zero_()
            reference.velocity_increment.zero_()
            wp.capture_launch(captured.graph)
            reference.step(incoming, want, control, None, dt, external_forces=external)
            np.testing.assert_array_equal(state.particle_f.numpy(), incoming.particle_f.numpy())
            np.testing.assert_array_equal(out.particle_q.numpy(), want.particle_q.numpy())
            np.testing.assert_array_equal(out.particle_qd.numpy(), want.particle_qd.numpy())

    def test_cpu_preparations_stay_eager(self):
        if wp.get_device(self.device).is_cuda:
            self.skipTest("CPU execution")
        dt, model, state, incoming, control, actual, reference, cache = self.fixture()
        # No graph exists on a CPU device: the start-up entry declines and issues nothing.
        with mock.patch.object(cache, "_evaluate", side_effect=AssertionError("evaluation")):
            self.assertFalse(cache.warm(state, control, None, dt))
        for _ in range(4):
            self.assertTrue(cache.prepare(state, control, None, dt))
        self.assertEqual((cache.preparations, cache.graph_captures, cache.graph_launches), (4, 0, 0))
        out, want = model.state(), model.state()
        actual.step(state, out, control, None, dt)
        reference.step(incoming, want, control, None, dt)
        np.testing.assert_array_equal(out.particle_q.numpy(), want.particle_q.numpy())
        np.testing.assert_array_equal(out.particle_qd.numpy(), want.particle_qd.numpy())


def _bits(array):
    return array.numpy().view(np.uint32).copy()


def _replacement(array, factor=1.25):
    """Another array of the same shape and type; floating values are scaled so that a stale reader differs."""
    copy = wp.clone(array)
    values = copy.numpy()
    if np.issubdtype(values.dtype, np.floating):
        copy.assign(values * values.dtype.type(factor))
    return copy


class HeldStartReplayTests(unittest.TestCase):
    """A preparation graph stands for the eager preparation of its binding and for no other."""

    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def setUp(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("Preparation graphs require CUDA")

    def fixture(self, *, mixed=True):
        source = TestPreparedFEMStep()
        source.device = self.device
        model, state, self.make_solver = source.fixture(mixed=mixed)
        solver = self.make_solver()
        solver.refresh_tangent(state.particle_q)
        return source.dt, model, state, model.control(), solver, HeldFEMStart(solver)

    def prepared(self, cache):
        return [_bits(array) for array in cache.owners[:4]]

    def assert_eager_values(self, cache, state, control, dt):
        """The prepared arrays hold the bits of a preparation that the same solver issues as written."""
        eager = HeldFEMStart(cache.solver)
        self.assertTrue(eager.prepare(state, control, None, dt))
        self.assertEqual((eager.graph_captures, eager.graph_launches), (0, 0))
        for actual, expected in zip(self.prepared(cache), self.prepared(eager), strict=True):
            np.testing.assert_array_equal(actual, expected)

    def establish(self, cache, state, control, dt):
        """Eager, captured and replayed preparation of one binding."""
        counters = (cache.graph_captures, cache.graph_launches)
        for captures, launches in ((0, 0), (1, 1), (1, 2)):
            self.assertTrue(cache.prepare(state, control, None, dt))
            self.assertEqual(
                (cache.graph_captures, cache.graph_launches), (counters[0] + captures, counters[1] + launches)
            )
            self.assert_eager_values(cache, state, control, dt)

    def test_replays_hold_the_bits_of_eager_preparations_over_alternating_accepted_states(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                dt, model, state, control, solver, cache = self.fixture(mixed=mixed)
                solver._held_start = cache
                reference = self.make_solver()
                following, incoming, expected = model.state(), model.state(), model.state()
                incoming.assign(state)
                load = wp.array(np.linspace(-0.01, 0.02, 24).reshape(8, 3), dtype=wp.vec3, device=self.device)
                steps = 8
                for _ in range(steps):
                    for each, positions in ((solver, state.particle_q), (reference, incoming.particle_q)):
                        each.refresh_tangent(positions)
                    start = [_bits(array) for array in (state.particle_q, state.particle_qd, state.particle_f)]
                    self.assertTrue(cache.prepare(state, control, None, dt))
                    self.assert_eager_values(cache, state, control, dt)
                    for array, bits in zip((state.particle_q, state.particle_qd, state.particle_f), start, strict=True):
                        np.testing.assert_array_equal(_bits(array), bits)
                    solver.velocity_increment.zero_()
                    reference.velocity_increment.zero_()
                    solver.step(state, following, control, None, dt, external_forces=load)
                    reference.step(incoming, expected, control, None, dt, external_forces=load)
                    np.testing.assert_array_equal(_bits(following.particle_q), _bits(expected.particle_q))
                    np.testing.assert_array_equal(_bits(following.particle_qd), _bits(expected.particle_qd))
                    state, following = following, state
                    incoming, expected = expected, incoming
                # Two accepted states alternate: each is prepared eagerly once and captured once.
                self.assertEqual(
                    (cache.preparations, cache.graph_captures, cache.graph_launches), (steps, 2, steps - 2)
                )
                self.assertEqual(cache.diagnostics()["cached_trial_assemblies"], steps)

    def test_warm_captures_both_start_states_ahead_and_attempts_then_only_launch(self):
        from newton_solver_suturebot._src.solvers.fem import held_start as held_module

        for mixed, scratch in ((False, True), (True, True), (False, False)):
            with self.subTest(mixed=mixed, scratch=scratch):
                dt, model, state, control, solver, cache = self.fixture(mixed=mixed)
                solver._held_start = cache
                other = model.state()
                other.assign(state)
                velocity = other.particle_qd.numpy()
                velocity[5] += np.float32(0.004)
                other.particle_qd.assign(velocity)
                buffers = (state, other)
                if scratch:
                    # The solver has its force scratch, so a preparation has a binding; none has a graph.
                    self.assertIsNotNone(cache._binding(state, control))
                else:
                    # A solver whose force scratch does not exist yet: without it no preparation
                    # has a binding that a graph could stand for. The first evaluation makes it.
                    solver._force_bufs.clear()
                    self.assertIsNone(cache._binding(state, control))
                self.assertEqual(len(cache._graphs), 0)
                start = [[_bits(a) for a in (b.particle_q, b.particle_qd, b.particle_f)] for b in buffers]
                scratch_at_capture = []
                capture = held_module.ScopedCapture

                def recording(*arguments, _capture=capture, _solver=solver, _seen=scratch_at_capture, **options):
                    _seen.append(all(_solver._force_bufs.get(name) is not None for name in held_module._FORCE_TERMS))
                    return _capture(*arguments, **options)

                with mock.patch.object(held_module, "ScopedCapture", recording):
                    for buffer in buffers:
                        self.assertTrue(cache.warm(buffer, control, None, dt))
                # The scratch existed before the first capture began; one graph per start state.
                self.assertEqual(scratch_at_capture, [True, True])
                self.assertEqual((cache.graph_captures, cache.graph_launches, cache.preparations), (2, 0, 0))
                self.assertEqual((cache.active, cache.signature), (False, None))
                self.assertEqual(len(cache._graphs), 2)
                self.assertEqual([graph.graph_exec is None for graph in cache._graphs.values()], [False, False])
                for buffer, bits in zip(buffers, start, strict=True):
                    for array, expected in zip(
                        (buffer.particle_q, buffer.particle_qd, buffer.particle_f), bits, strict=True
                    ):
                        np.testing.assert_array_equal(_bits(array), expected)
                # A second call finds both graphs.
                with mock.patch.object(held_module, "ScopedCapture", side_effect=AssertionError("capture")):
                    for buffer in buffers:
                        self.assertTrue(cache.warm(buffer, control, None, dt))
                    # The attempts of a run: every preparation launches the graph of its start state.
                    count = 6
                    for index in range(count):
                        buffer = buffers[index % 2]
                        self.assertTrue(cache.prepare(buffer, control, None, dt))
                        self.assertEqual(
                            (cache.graph_captures, cache.graph_launches, cache.preparations), (2, index + 1, index + 1)
                        )
                        self.assertTrue(cache.matches(buffer, dt, control))
                        self.assert_eager_values(cache, buffer, control, dt)
                        cache.invalidate()
                for buffer, bits in zip(buffers, start, strict=True):
                    for array, expected in zip(
                        (buffer.particle_q, buffer.particle_qd, buffer.particle_f), bits, strict=True
                    ):
                        np.testing.assert_array_equal(_bits(array), expected)

    def test_warm_declines_where_a_preparation_is_not_served_and_refuses_another_timestep(self):
        dt, model, state, control, solver, cache = self.fixture(mixed=False)
        solver._held_start = cache
        # Contacts are not a supported input of the held start; nothing is issued for them.
        with mock.patch.object(cache, "_evaluate", side_effect=AssertionError("evaluation")):
            self.assertFalse(cache.warm(state, control, object(), dt))
            self.assertFalse(cache.warm(state, None, None, dt))
            with self.assertRaisesRegex(ValueError, "assembled FEM tangent"):
                cache.warm(state, control, None, 0.5 * dt)
            with ScopedCapture(device=self.device):
                cache.predictor.zero_()
                with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                    cache.warm(state, control, None, dt)
        self.assertEqual((cache.graph_captures, len(cache._graphs)), (0, 0))

    def test_timestep_changes_replay_the_graphs_of_both_start_states_without_recapture(self):
        # A rejected substep is retried with half the timestep and later steps return to the
        # base step. The preparation reads the timestep from the solver's step time, so the two
        # graphs of the two alternating accepted states serve every timestep: the first one is
        # captured at the halved step and replays at the base step, a third and back.
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                base, model, state, control, solver, cache = self.fixture(mixed=mixed)
                solver._held_start = cache
                reference = self.make_solver()
                following, incoming, expected = model.state(), model.state(), model.state()
                load = wp.array(np.linspace(-0.01, 0.02, 24).reshape(8, 3), dtype=wp.vec3, device=self.device)
                steps = (base, base, base / 2, base / 2, base, base, base / 3, base / 2, base, base)
                graphs = None
                for index, dt in enumerate(steps):
                    # The plain step of a second solver starts from the bits of this start state.
                    incoming.assign(state)
                    for each, positions in ((solver, state.particle_q), (reference, incoming.particle_q)):
                        each.refresh_tangent(positions, dt)
                    self.assertTrue(cache.prepare(state, control, None, dt))
                    self.assert_eager_values(cache, state, control, dt)
                    # The impulse h * sum(f) is not zero: a graph that kept the timestep of its
                    # capture would differ from the eager preparation above by the ratio of the two.
                    self.assertTrue(np.any(cache.state.particle_f.numpy()))
                    # Each start state: one eager preparation, one capture, replays from then on.
                    self.assertEqual(
                        (cache.preparations, cache.graph_captures, cache.graph_launches),
                        (index + 1, min(max(index - 1, 0), 2), max(index - 1, 0)),
                    )
                    self.assertEqual(solver.step_time.value, dt)
                    np.testing.assert_array_equal(solver.step_time.seconds.numpy(), [np.float32(dt)])
                    current = list(cache._graphs.values())
                    if index == 3:
                        graphs = current
                        self.assertEqual(len(graphs), 2)
                        self.assertNotIn(None, graphs)
                    elif index > 3:
                        self.assertEqual({id(graph) for graph in current}, {id(graph) for graph in graphs})
                    solver.velocity_increment.zero_()
                    reference.velocity_increment.zero_()
                    solver.step(state, following, control, None, dt, external_forces=load)
                    reference.step(incoming, expected, control, None, dt, external_forces=load)
                    for name in ("particle_q", "particle_qd"):
                        actual, wanted = getattr(following, name).numpy(), getattr(expected, name).numpy()
                        if mixed:
                            # The hexahedron, triangle and spring add to the nodes of the tetrahedron
                            # with atomic additions, whose order differs between two launches: the
                            # step through the held start and the plain step agree to the last bits
                            # only (measured: 6 units in the last place of an entry over 200 steps).
                            # The bound is 16 units in the last place of the largest entry.
                            np.testing.assert_allclose(
                                actual,
                                wanted,
                                rtol=0,
                                atol=16 * np.spacing(np.abs(wanted).max()),
                                err_msg=f"step {index} {name}",
                            )
                        else:
                            np.testing.assert_array_equal(
                                actual.view(np.uint32), wanted.view(np.uint32), err_msg=f"step {index} {name}"
                            )
                    state, following = following, state
                self.assertEqual(cache.diagnostics()["cached_trial_assemblies"], len(steps))

    def test_values_written_in_place_are_read_by_the_kept_graph(self):
        dt, model, state, control, solver, cache = self.fixture()
        solver.enable_tet_activations = True
        self.establish(cache, state, control, dt)
        positions, velocities = state.particle_q.numpy(), state.particle_qd.numpy()
        positions[4] += (0.002, -0.001, 0.003)
        velocities[5] += (0.01, 0.02, -0.01)

        def refresh_topology():
            materials = model.tet_materials.numpy()
            materials[:, :2] *= 1.5
            model.tet_materials.assign(materials)
            solver.refresh_topology(state.particle_q)

        edits = {
            "start positions": lambda: state.particle_q.assign(positions),
            "start velocities": lambda: state.particle_qd.assign(velocities),
            "refreshed tangent": lambda: solver.refresh_tangent(state.particle_q),
            "topology within fixed storage": refresh_topology,
            "gravity": lambda: model.gravity.assign(np.asarray([[0.1, -0.2, -0.5]], dtype=np.float32)),
            "masses": lambda: model.particle_mass.assign(1.5 * model.particle_mass.numpy()),
            "activations": lambda: control.tet_activations.fill_(0.25),
            "fixed reference": lambda: solver._fixed_particles.reference.assign(
                1.1 * solver._fixed_particles.reference.numpy()
            ),
            "solver reset": lambda: solver.reset(state),
        }
        counters = (cache.graph_captures, cache.graph_launches)
        for replays, (name, edit) in enumerate(edits.items(), start=1):
            with self.subTest(edit=name):
                before = self.prepared(cache)
                edit()
                self.assertTrue(cache.prepare(state, control, None, dt))
                self.assertEqual((cache.graph_captures, cache.graph_launches), (counters[0], counters[1] + replays))
                self.assert_eager_values(cache, state, control, dt)
                if name != "solver reset":
                    self.assertFalse(
                        all(np.array_equal(old, new) for old, new in zip(before, self.prepared(cache), strict=True))
                    )

    def test_every_replaced_array_scalar_and_force_law_is_prepared_eagerly_before_its_capture(self):
        dt, model, state, control, solver, cache = self.fixture()
        control.tet_activations.fill_(0.2)
        solver.spring_rest_direction.fill_(wp.vec3(1.0, 0.0, 0.0))
        self.establish(cache, state, control, dt)
        fixed, matrix = solver._fixed_particles, solver.A_bsr
        scalars = (
            ("linear_damping", 0.7),
            ("linear_damping", 0.0),
            ("min_stretch", 0.9),
            ("max_stretch", 1.1),
            ("enable_tet_activations", True),
            ("torque_stiffness", 0.05),
            ("torque_damping", 0.01),
        )
        arrays = (
            (state, ("particle_q", "particle_qd")),
            (cache.state, ("particle_q", "particle_qd", "particle_f")),
            (cache, ("predictor",)),
            (control, ("tet_activations",)),
            (model, ("gravity", "particle_mass", "particle_flags", "tet_indices", "tet_poses", "tet_materials")),
            (model, ("tri_indices", "tri_poses", "tri_materials")),
            (model, ("spring_indices", "spring_rest_length", "spring_stiffness", "spring_damping")),
            (solver, ("hex_indices", "hex_inv_J0", "hex_det_J0_w", "hex_materials", "spring_rest_direction")),
            (matrix, ("offsets", "columns", "values")),
            (fixed, ("indices", "reference")),
        )
        cases = 0
        for name, value in scalars:
            with self.subTest(scalar=name, value=value):
                before = self.prepared(cache)
                setattr(solver, name, value)
                self.establish(cache, state, control, dt)
                self.assertFalse(
                    all(np.array_equal(old, new) for old, new in zip(before, self.prepared(cache), strict=True))
                )
                cases += 1
        for owner, names in arrays:
            for name in names:
                with self.subTest(owner=type(owner).__name__, array=name):
                    setattr(owner, name, _replacement(getattr(owner, name)))
                    self.establish(cache, state, control, dt)
                    cases += 1
        for name in ("surface", "spring", "tet", "hex", "tri", "ground", "gravity", "damping"):
            with self.subTest(force_scratch=name):
                solver._force_bufs[name] = _replacement(solver._force_bufs[name])
                self.establish(cache, state, control, dt)
                cases += 1
        with self.subTest(change="timestep"):
            # The timestep is device data of the solver's step time, not a part of the binding:
            # the kept graph replays at the new one with the bits of an eager preparation.
            counters = (cache.graph_captures, cache.graph_launches)
            before = self.prepared(cache)
            dt *= 0.5
            solver.refresh_tangent(state.particle_q, dt)
            for replays in (1, 2):
                self.assertTrue(cache.prepare(state, control, None, dt))
                self.assertEqual((cache.graph_captures, cache.graph_launches), (counters[0], counters[1] + replays))
                self.assert_eager_values(cache, state, control, dt)
            self.assertFalse(np.array_equal(before[2], self.prepared(cache)[2]))
        with self.subTest(change="force law"):

            def weightless(self, model):
                return self._forces("gravity")

            before = self.prepared(cache)
            with mock.patch.object(SolverFEM, "eval_gravity_forces", weightless):
                self.establish(cache, state, control, dt)
                self.assertFalse(np.array_equal(before[2], self.prepared(cache)[2]))
            self.establish(cache, state, control, dt)
            np.testing.assert_array_equal(before[2], self.prepared(cache)[2])
            cases += 2
        with self.subTest(change="missing force scratch"):
            del solver._force_bufs["tet"]
            counters = (cache.graph_captures, cache.graph_launches)
            self.assertTrue(cache.prepare(state, control, None, dt))
            self.assertEqual((cache.graph_captures, cache.graph_launches), counters)
            self.assert_eager_values(cache, state, control, dt)
            self.establish(cache, state, control, dt)
            cases += 1
        self.assertEqual(cache.graph_captures, 1 + cases)

    def test_start_states_that_do_not_return_are_never_captured_and_an_evicted_one_starts_over(self):
        dt, model, state, control, solver, cache = self.fixture()
        self.establish(cache, state, control, dt)
        counters = (cache.graph_captures, cache.graph_launches)
        # Live states have distinct storage; a freed one would hand its storage to the next.
        others = [model.state() for _ in range(_GRAPH_LIMIT)]
        for other in others:
            other.assign(state)
            self.assertTrue(cache.prepare(other, control, None, dt))
            self.assert_eager_values(cache, other, control, dt)
        self.assertEqual((cache.graph_captures, cache.graph_launches), counters)
        self.establish(cache, state, control, dt)

    def test_a_kept_graph_is_not_launched_by_an_unsupported_or_rejected_preparation(self):
        dt, model, state, control, solver, cache = self.fixture()
        self.establish(cache, state, control, dt)
        counters = (cache.preparations, cache.graph_captures, cache.graph_launches)
        before = self.prepared(cache)
        with wp.Tape():
            self.assertFalse(cache.prepare(state, control, None, dt))
        forces = state.particle_f
        state.particle_f = wp.clone(forces, requires_grad=True)
        self.assertFalse(cache.prepare(state, control, None, dt))
        state.particle_f = forces
        with mock.patch.object(solver, "_accumulate_forces", wraps=solver._accumulate_forces):
            self.assertFalse(cache.prepare(state, control, None, dt))
        with self.assertRaisesRegex(ValueError, "assembled FEM tangent"):
            cache.prepare(state, control, None, 0.5 * dt)
        scratch = wp.zeros(1, dtype=int, device=self.device)
        with ScopedCapture(device=self.device):
            scratch.zero_()
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                cache.prepare(state, control, None, dt)
        self.assertFalse(cache.active)
        self.assertEqual((cache.preparations, cache.graph_captures, cache.graph_launches), counters)
        for old, new in zip(before, self.prepared(cache), strict=True):
            np.testing.assert_array_equal(old, new)
        self.assertTrue(cache.prepare(state, control, None, dt))
        self.assertEqual(cache.graph_launches, counters[2] + 1)
        self.assert_eager_values(cache, state, control, dt)


if __name__ == "__main__":
    unittest.main()
