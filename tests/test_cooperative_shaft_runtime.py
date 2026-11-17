# SPDX-License-Identifier: Apache-2.0
"""Fresh geometry reuse cannot cache forces, duals or a rejected physical state."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from _cooperative_projection_trace import traced_projection
from test_combined_contact_preconditioner import native_embedded
from test_needle_insertion_example import _example
from test_needle_tissue_coupler import _native_state
from test_needle_tunnel_coupler import _STEP, _arrays, _prepare

from newton_solver_suturebot._src.solvers.coupled.interface.insertion import _shaft_projection
from newton_solver_suturebot._src.solvers.coupled.interface.shaft_projection import (
    ProjectionBinding,
    prepare_projection,
    prepare_queries,
    projection_binding,
)
from newton_solver_suturebot._src.utils.capture import ScopedCapture

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def original_packet(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body: int,
    radius: float,
    samples: wp.array[wp.vec3],
    result: wp.array2d[float],
):
    i = wp.tid()
    point = wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
    closest, tangent, distance, covered = _shaft_projection(point, body_q[body], samples, radius)
    for axis in range(3):
        result[i, axis] = closest[axis]
        result[i, axis + 3] = tangent[axis]
    result[i, 6] = distance
    result[i, 7] = float(covered)


@wp.kernel(enable_backward=False)
def original_material_branches(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body: int,
    radius: float,
    samples: wp.array[wp.vec3],
    branches: wp.array2d[int],
):
    i = wp.tid()
    point = wp.vec3(0.0)
    for j in range(4):
        point += bary[i][j] * q[nodes[i][j]]
    closest, tangent, distance, covered, branch = traced_projection(point, body_q[body], samples, radius)
    for axis in range(6):
        branches[i, axis] = branch[axis]


def fixture(device=DEVICE, *, capacity=2, count=2):
    """Two track samples in storage of ``capacity`` rows, of which ``count`` are live."""

    def array(a, dtype):
        return wp.array(a, dtype=dtype, device=device)

    # Straight authored needle gives an exact geometric reference, including
    # endpoint ties. Every material node contributes to the nontrivial points.
    rows, weights = np.zeros((capacity, 4), np.int32), np.zeros((capacity, 4), np.float32)
    rows[:2], weights[:2] = [[0, 1, 2, 3], [1, 2, 3, 0]], [[0.125, 0.25, 0.125, 0.5], [0.25, 0.25, 0.25, 0.25]]
    nodes = array(rows, wp.vec4i)
    bary = array(weights, wp.vec4)
    q = array([[-0.002, 0.0002, 0], [0.002, 0.0003, 0], [0.001, 0.0004, 0], [0, 0.0001, 0]], wp.vec3)
    samples = array([[-0.01, 0, 0], [0, 0, 0], [0.01, 0, 0]], wp.vec3)
    interaction = SimpleNamespace(
        device=wp.get_device(device),
        _track_nodes=nodes,
        _track_bary=bary,
        _samples=samples,
        track_count=count,
        track_capacity=capacity,
        _live=array([0, count, 0], int),
        cooperative_shaft_projection=True,
    )
    state = SimpleNamespace(
        particle_q=q,
        particle_qd=wp.zeros_like(q),
        body_q=array([wp.transform_identity()], wp.transform),
        body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=device),
    )
    return interaction, state


def reference(binding):
    out = wp.empty_like(binding.result)
    wp.launch(
        original_packet,
        binding.capacity,
        [binding.nodes, binding.bary, binding.q, binding.body_q, binding.body, binding.radius, binding.samples, out],
        device=binding.device,
    )
    return out.numpy()


class ProjectionRuntimeContracts(unittest.TestCase):
    def test_specialized_kernels_change_only_projection_lookup(self):
        import newton_solver_suturebot._src.solvers.coupled as coupled

        root = Path(coupled.__file__).parent
        originals = [
            ("interface/insertion.py", "_shaft_forces", "shaft_forces_projected"),
            ("interface/tunnel_alm.py", "_tunnel_forces", "tunnel_forces_projected"),
            ("acceleration/shaft_preconditioner.py", "prepare_anchors", "prepare_anchors_projected"),
            ("acceleration/shaft_preconditioner.py", "prepare_alm_anchors", "prepare_alm_anchors_projected"),
        ]
        specialized = ast.parse((root / "interface/shaft_projected_kernels.py").read_text())

        class Normalize(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in ("_shaft_projection", "read_projection"):
                    return ast.Name(id="PROJECTION_RESULT", ctx=ast.Load())
                return self.generic_visit(node)

        for filename, name, other in originals:
            original = next(
                n
                for n in ast.parse((root / filename).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            candidate = next(n for n in specialized.body if isinstance(n, ast.FunctionDef) and n.name == other)
            self.assertEqual(
                [ast.dump(Normalize().visit(n), include_attributes=False) for n in original.body],
                [ast.dump(Normalize().visit(n), include_attributes=False) for n in candidate.body],
            )

    def test_cpu_and_gradient_paths_do_not_bind(self):
        # With samples and with an empty track in a reserve: the original projection stays in both.
        for options in ({}, {"capacity": 6, "count": 0}):
            with self.subTest(**options):
                interaction, state = fixture("cpu", **options)
                self.assertIsNone(projection_binding(interaction, state, 0, 0.00028))
                self.assertFalse(hasattr(interaction, "_cooperative_shaft_binding"))
                self.assertFalse(hasattr(interaction, "_cooperative_shaft_bindings"))
                if wp.get_device(DEVICE).is_cuda:
                    interaction, state = fixture(**options)
                    with wp.Tape():
                        self.assertIsNone(projection_binding(interaction, state, 0, 0.00028))
                    state.particle_qd = wp.clone(state.particle_qd, requires_grad=True)
                    self.assertIsNone(projection_binding(interaction, state, 0, 0.00028))
                    self.assertFalse(hasattr(interaction, "_cooperative_shaft_binding"))
                    # Storage that does not have the rows of the track reserve is not bound either.
                    interaction, state = fixture(**options)
                    interaction.track_capacity += 1
                    self.assertIsNone(projection_binding(interaction, state, 0, 0.00028))

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA cooperative query")
    def test_binding_exists_from_an_empty_track_and_runs_over_the_reserve(self):
        interaction, state = fixture(capacity=6, count=0)
        binding = projection_binding(interaction, state, 0, 0.00028)
        self.assertIsNotNone(binding)
        self.assertIs(interaction._cooperative_shaft_binding, binding)
        self.assertEqual(binding.capacity, 6)
        for name in ("queries", "poses", "radii", "result", "branches", "initialized", "generations", "dirty"):
            self.assertEqual(getattr(binding, name).shape[0], 6, name)
        self.assertTrue(any(owner is interaction._live for owner in binding.owners))
        sentinel = -7.0
        binding.result.fill_(sentinel)
        sizes = []
        launch = wp.launch

        def record(kernel, dim, *arguments, **options):
            sizes.append((kernel.key, dim))
            return launch(kernel, dim, *arguments, **options)

        with patch.object(wp, "launch", side_effect=record):
            self.assertIs(prepare_projection(interaction, state, 0, 0.00028), binding)
        # A batch is these two launches and no other: both kernels run over the reserve, the
        # projection in whole blocks of 256 threads, and nothing is issued for the arc table.
        self.assertEqual(sizes, [("prepare_queries", 6), ("cooperative_projection", 256)])
        # An empty track: nothing is marked, projected or counted, and no result row is written.
        np.testing.assert_array_equal(binding.counts.numpy(), [0, 0])
        np.testing.assert_array_equal(binding.dirty.numpy(), 0)
        np.testing.assert_array_equal(binding.initialized.numpy(), 0)
        np.testing.assert_array_equal(binding.result.numpy(), sentinel)
        with ScopedCapture(device=interaction.device) as capture:
            self.assertIs(prepare_projection(interaction, state, 0, 0.00028), binding)
        key = binding.key
        # Two samples become live: the count is device data, so neither the binding nor its key
        # changes, and the graph captured at the empty track projects them.
        interaction.track_count = 2
        interaction._live.assign([0, 2, 0])
        self.assertIs(projection_binding(interaction, state, 0, 0.00028), binding)
        self.assertEqual(ProjectionBinding.binding_key(interaction, state, 0, 0.00028), key)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(binding.counts.numpy(), [2, 0])
        np.testing.assert_allclose(binding.result.numpy()[:2], reference(binding)[:2], rtol=4e-7, atol=2e-10)
        np.testing.assert_array_equal(binding.result.numpy()[2:], sentinel)
        np.testing.assert_array_equal(binding.initialized.numpy(), [1, 1, 0, 0, 0, 0])
        # Unchanged inputs are fresh at the next replay; the rows beyond the count stay untouched.
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(binding.counts.numpy(), [2, 2])
        np.testing.assert_array_equal(binding.dirty.numpy(), 0)
        np.testing.assert_array_equal(binding.result.numpy()[2:], sentinel)
        self.assertEqual(interaction._cooperative_shaft_rebinds, 0)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA cooperative query")
    def test_interaction_keeps_a_binding_for_the_candidate_and_both_state_buffers(self):
        interaction, state = fixture(capacity=6)

        def other():
            return SimpleNamespace(
                particle_q=wp.clone(state.particle_q),
                particle_qd=wp.zeros_like(state.particle_q),
                body_q=wp.clone(state.body_q),
                body_qd=wp.zeros_like(state.body_qd),
            )

        def bind(target):
            return projection_binding(interaction, target, 0, 0.00028)

        states = [state, other(), other()]
        bindings = [bind(target) for target in states]
        self.assertEqual(len({id(binding) for binding in bindings}), 3)
        self.assertEqual(len({binding.key for binding in bindings}), 3)
        # Alternating between the three never rebinds: each key keeps its packet and its counters' owner.
        for _ in range(3):
            for target, binding in zip(states, bindings, strict=True):
                self.assertIs(bind(target), binding)
                self.assertIs(interaction._cooperative_shaft_binding, binding)
        self.assertEqual(interaction._cooperative_shaft_rebinds, 0)
        self.assertEqual(list(interaction._cooperative_shaft_bindings.values()), bindings)
        self.assertEqual(len({id(binding.counts) for binding in bindings}), 1)
        # Inside a capture a bound key is served and an unbound one raises: its graph would keep
        # the original projection for every replay. Nothing is made or dropped there.
        fourth = other()
        with ScopedCapture(device=interaction.device):
            self.assertIs(bind(states[1]), bindings[1])
            with self.assertRaisesRegex(RuntimeError, "no projection is bound"):
                bind(fourth)
        self.assertEqual(len(interaction._cooperative_shaft_bindings), 3)
        self.assertEqual(interaction._cooperative_shaft_rebinds, 0)
        # On another stream than the bindings' the original projection stays, in a capture as outside one.
        with wp.ScopedStream(wp.Stream(device=interaction.device)):
            self.assertIsNone(bind(fourth))
            with ScopedCapture(device=interaction.device):
                self.assertIsNone(bind(fourth))
        self.assertEqual(len(interaction._cooperative_shaft_bindings), 3)
        self.assertEqual(interaction._cooperative_shaft_rebinds, 0)
        # A fourth key outside a capture takes the place of the least recently returned binding.
        self.assertIs(bind(states[0]), bindings[0])
        self.assertIs(bind(states[1]), bindings[1])
        new = bind(fourth)
        self.assertEqual(interaction._cooperative_shaft_rebinds, 1)
        self.assertEqual(list(interaction._cooperative_shaft_bindings.values()), [bindings[0], bindings[1], new])
        self.assertIsNot(bind(states[2]), bindings[2])
        self.assertEqual(interaction._cooperative_shaft_rebinds, 2)
        # Replaced track storage: every binding that addresses the old storage goes at the next new key.
        kept = list(interaction._cooperative_shaft_bindings.values())
        interaction._track_nodes = wp.clone(interaction._track_nodes)
        replacement = bind(states[0])
        self.assertTrue(all(binding.stale(interaction) for binding in kept))
        self.assertFalse(replacement.stale(interaction))
        self.assertEqual(list(interaction._cooperative_shaft_bindings.values()), [replacement])
        self.assertEqual(interaction._cooperative_shaft_rebinds, 5)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA cooperative query")
    def test_binding_returned_last_is_served_again_without_a_key(self):
        interaction, state = fixture(capacity=6)
        binding = projection_binding(interaction, state, 0, 0.00028)
        keys = []
        form = ProjectionBinding.binding_key

        def counted(*arguments):
            keys.append(arguments)
            return form(*arguments)

        def bind(radius=0.00028):
            return projection_binding(interaction, state, 0, radius)

        with patch.object(ProjectionBinding, "binding_key", side_effect=counted):
            # The call of every substep: the same arrays and settings. No key is formed.
            for _ in range(3):
                self.assertIs(bind(), binding)
            # Another radius is another key and another binding: one key for the lookup and one for
            # the new binding. Asked for again it is the binding returned last, without a key. The
            # first binding is then found by its key, which is the third and last key formed.
            other_radius = float(np.float32(0.00028))
            self.assertIs(bind(other_radius), bind(other_radius))
            self.assertIs(prepare_projection(interaction, state, 0, 0.00028), binding)
            self.assertEqual(len(keys), 3)

    def test_binding_serves_the_arrays_and_settings_of_its_key_and_no_other(self):
        # Host logic of the binding returned last; the binding is constructed directly, as on CUDA
        # projection_binding does.
        interaction, state = fixture("cpu", capacity=6)
        binding = ProjectionBinding(interaction, state, 0, 0.00028)
        self.assertEqual(binding.key, ProjectionBinding.binding_key(interaction, state, 0, 0.00028))
        self.assertTrue(binding.serves(interaction, state, 0, 0.00028))
        self.assertTrue(binding.serves(interaction, state, np.int32(0), np.float64(0.00028)))
        # Other settings are another key.
        self.assertFalse(binding.serves(interaction, state, 1, 0.00028))
        self.assertFalse(binding.serves(interaction, state, 0, 0.0003))
        self.assertFalse(binding.serves(interaction, state, 0, float(np.float32(0.00028))))
        # Another state object with the same arrays is the same key; another array in a place of the key is not.
        same = SimpleNamespace(**vars(state))
        self.assertTrue(binding.serves(interaction, same, 0, 0.00028))
        self.assertEqual(binding.key, ProjectionBinding.binding_key(interaction, same, 0, 0.00028))
        for name in ("particle_q", "body_q"):
            other = SimpleNamespace(**vars(state))
            setattr(other, name, wp.clone(getattr(state, name)))
            self.assertFalse(binding.serves(interaction, other, 0, 0.00028), name)
            self.assertNotEqual(binding.key, ProjectionBinding.binding_key(interaction, other, 0, 0.00028), name)
        # An array that is no part of the key does not matter to the key; projection_binding checks it in every call.
        other = SimpleNamespace(**vars(state))
        other.particle_qd = wp.clone(state.particle_qd)
        self.assertTrue(binding.serves(interaction, other, 0, 0.00028))
        self.assertEqual(binding.key, ProjectionBinding.binding_key(interaction, other, 0, 0.00028))
        # Replaced storage of the interaction: stale, another key, not served.
        for name in ("_track_nodes", "_track_bary", "_live", "_samples"):
            saved = getattr(interaction, name)
            setattr(interaction, name, wp.clone(saved))
            self.assertTrue(binding.stale(interaction), name)
            self.assertFalse(binding.serves(interaction, state, 0, 0.00028), name)
            self.assertNotEqual(binding.key, ProjectionBinding.binding_key(interaction, state, 0, 0.00028), name)
            setattr(interaction, name, saved)
        self.assertFalse(binding.stale(interaction))
        self.assertTrue(binding.serves(interaction, state, 0, 0.00028))

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA cooperative query")
    def test_capture_without_a_bound_projection_raises_at_any_track_count(self):
        # The track count is no part of a graph key, so a graph captured without a binding would
        # keep the original projection after the first sample as well. It is refused instead.
        for options in ({"capacity": 6, "count": 0}, {"capacity": 6}):
            with self.subTest(**options):
                interaction, state = fixture(**options)
                with ScopedCapture(device=interaction.device):
                    with self.assertRaisesRegex(RuntimeError, "no projection is bound"):
                        prepare_projection(interaction, state, 0, 0.00028)
                    with self.assertRaisesRegex(RuntimeError, "no projection is bound"):
                        projection_binding(interaction, state, 0, 0.00028)
                # The refused capture made no binding and counted none.
                self.assertEqual(interaction._cooperative_shaft_bindings, {})
                self.assertEqual(interaction._cooperative_shaft_rebinds, 0)
                self.assertFalse(hasattr(interaction, "_cooperative_shaft_binding"))
                # Bound outside a capture, the same call is served inside one.
                binding = projection_binding(interaction, state, 0, 0.00028)
                with ScopedCapture(device=interaction.device) as capture:
                    self.assertIs(prepare_projection(interaction, state, 0, 0.00028), binding)
                interaction.track_count = 2
                interaction._live.assign([0, 2, 0])
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(binding.result.numpy()[:2], reference(binding)[:2], rtol=4e-7, atol=2e-10)
                # Without the opt-in, on the CPU and with a gradient state a capture keeps the original projection.
                interaction.cooperative_shaft_projection = False
                with ScopedCapture(device=interaction.device):
                    self.assertIsNone(projection_binding(interaction, state, 0, 0.00028))
                interaction.cooperative_shaft_projection = True
                gradient_force = wp.empty(4, dtype=wp.vec3, device=interaction.device, requires_grad=True)
                with ScopedCapture(device=interaction.device):
                    self.assertIsNone(projection_binding(interaction, state, 0, 0.00028, operands=(gradient_force,)))

    def test_arc_table_is_built_with_the_binding_and_a_batch_issues_nothing_for_it(self):
        # Host and CPU kernels only: the binding is constructed directly, as on CUDA projection_binding does.
        interaction, state = fixture("cpu", capacity=6, count=0)
        issued = []
        launch = wp.launch

        def record(kernel, *arguments, **options):
            issued.append(kernel.key)
            return launch(kernel, *arguments, **options)

        with patch.object(wp, "launch", side_effect=record):
            binding = ProjectionBinding(interaction, state, 0, 0.00028)
        # The one-thread refresh of the table runs once, when the binding is made.
        self.assertEqual(issued, ["refresh_table"])
        table = binding.tables.table

        def flags():
            return [int(array.numpy()[0]) for array in (table.initialized, table.valid, table.generation)]

        self.assertEqual(flags(), [1, 1, 1])
        np.testing.assert_allclose(table.arcs.numpy(), [0.0, 0.01, 0.02], rtol=1e-6)
        # A batch issues the two query kernels with that table and nothing for the table itself.
        # Recorded without launching: the projection kernel is written for CUDA lanes.
        with patch.object(wp, "launch") as batch:
            binding.ensure()
        self.assertEqual(
            [call.args[0].key for call in batch.call_args_list], ["prepare_queries", "cooperative_projection"]
        )
        for call in batch.call_args_list:
            self.assertTrue(any(argument is table for argument in call.args[2]))
        self.assertEqual(flags(), [1, 1, 1])
        # Unchanged samples leave the generation; an edit in place is taken up by refresh_table.
        del issued[:]
        with patch.object(wp, "launch", side_effect=record):
            binding.refresh_table()
        self.assertEqual((issued, flags()), (["refresh_table"], [1, 1, 1]))
        samples = interaction._samples.numpy()
        samples[2, 0] = 0.03
        interaction._samples.assign(samples)
        binding.refresh_table()
        self.assertEqual(flags(), [1, 1, 2])
        np.testing.assert_allclose(table.arcs.numpy(), [0.0, 0.01, 0.04], rtol=1e-6)
        np.testing.assert_array_equal(table.samples.numpy(), samples)

    def test_real_scene_enables_geometry_only_after_interaction_construction(self):
        from test_grasp_scene import _scene

        with _scene() as (scene, _):
            self.assertTrue(scene.interaction.cooperative_shaft_projection)
            self.assertIsNone(getattr(scene.interaction, "_cooperative_shaft_binding", None))
            self.assertEqual(scene.elapsed, 0)

    def test_point_pose_and_source_generation_freshness_on_cpu(self):
        interaction, state = fixture("cpu")
        # CPU exercises the exact freshness kernels without executing CUDA shuffle.
        binding = ProjectionBinding(interaction, state, 0, 0.00028)

        def prepare():
            table = binding.tables.refresh()
            wp.launch(
                prepare_queries,
                binding.capacity,
                [
                    binding.nodes,
                    binding.bary,
                    binding.live,
                    binding.q,
                    binding.body_q,
                    binding.body,
                    binding.radius,
                    table,
                    binding.queries,
                    binding.poses,
                    binding.radii,
                    binding.initialized,
                    binding.generations,
                    binding.dirty,
                    binding.counts,
                ],
                device="cpu",
            )
            return binding.dirty.numpy()

        np.testing.assert_array_equal(prepare(), 1)
        binding.initialized.fill_(1)
        binding.generations.fill_(int(binding.tables.table.generation.numpy()[0]))
        np.testing.assert_array_equal(prepare(), 0)
        original = state.particle_q.numpy().copy()
        q = original.copy()
        q[0, 1] += 1e-5
        state.particle_q.assign(q)
        np.testing.assert_array_equal(prepare(), 1)
        np.testing.assert_array_equal(prepare(), 0)  # current bits have been recorded
        state.particle_q.assign(original)
        np.testing.assert_array_equal(prepare(), 1)
        samples = interaction._samples.numpy()
        samples[1, 2] = 1e-5
        interaction._samples.assign(samples)
        np.testing.assert_array_equal(prepare(), 1)
        binding.generations.fill_(int(binding.tables.table.generation.numpy()[0]))
        np.testing.assert_array_equal(prepare(), 0)
        pose = state.body_q.numpy()
        pose[0, 0] += 1e-5
        state.body_q.assign(pose)
        np.testing.assert_array_equal(prepare(), 1)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA cooperative query")
    def test_force_pc_reuse_and_capture_detect_inplace_edits_and_replacements(self):
        interaction, state = fixture()
        binding = prepare_projection(interaction, state, 0, 0.00028)
        np.testing.assert_allclose(binding.result.numpy(), reference(binding), rtol=4e-7, atol=2e-10)
        np.testing.assert_array_equal(binding.counts.numpy(), [2, 0])
        with ScopedCapture(device=interaction.device) as capture:
            binding.ensure()
        for kind in ["same", "q", "pose", "sample", "bary", "rollback"]:
            if kind == "q":
                values = state.particle_q.numpy()
                values[0, 1] += 1e-5
                state.particle_q.assign(values)
            if kind == "pose":
                values = state.body_q.numpy()
                values[0, 2] += 0.00003
                state.body_q.assign(values)
            if kind == "sample":
                values = interaction._samples.numpy()
                values[1, 0] += 0.00001
                interaction._samples.assign(values)
                # The table is built with the binding: an edit of the samples in place is announced,
                # outside the capture, and the graph captured before projects every row again.
                binding.refresh_table()
            if kind == "bary":
                interaction._track_bary.assign([[0.25, 0.25, 0.25, 0.25], [0.125, 0.125, 0.25, 0.5]])
            if kind == "rollback":
                state.body_q.assign([wp.transform_identity()])
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(binding.result.numpy(), reference(binding), rtol=4e-7, atol=2e-10, err_msg=kind)
        self.assertGreater(binding.counts.numpy()[0], 2)
        self.assertGreaterEqual(binding.counts.numpy()[1], 2)
        with wp.ScopedStream(wp.Stream(device=interaction.device)):
            self.assertIsNone(projection_binding(interaction, state, 0, 0.00028))
        self.assertIs(interaction._cooperative_shaft_binding, binding)
        gradient_force = wp.empty(4, dtype=wp.vec3, device=interaction.device, requires_grad=True)
        self.assertIsNone(projection_binding(interaction, state, 0, 0.00028, operands=(gradient_force,)))
        old = binding.result.numpy().copy()
        interaction._samples = wp.clone(interaction._samples)
        replacement = prepare_projection(interaction, state, 0, 0.00028)
        self.assertIsNot(binding, replacement)
        self.assertIsNot(binding.tables.table, replacement.tables.table)
        wp.capture_launch(capture.graph)  # old graph keeps all old packet owners
        np.testing.assert_array_equal(binding.result.numpy(), old)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "CUDA material-point interpolation")
    def test_curved_material_points_keep_original_nearest_and_arc_branches(self):
        rng = np.random.default_rng(778)
        interaction, state = fixture()
        theta = np.linspace(-1.2, 1.2, 193)
        samples = np.column_stack((0.01 * np.cos(theta), 0.01 * np.sin(theta), np.zeros_like(theta)))
        interaction._samples = wp.array(samples, dtype=wp.vec3, device=DEVICE)
        n = 128
        points = samples[rng.integers(0, len(samples), n)] + rng.normal(0, 0.0003, (n, 3))
        vertices = (points[:, None, :] + rng.normal(0, 0.0001, (n, 4, 3))).reshape(-1, 3)
        state.particle_q = wp.array(vertices, dtype=wp.vec3, device=DEVICE)
        state.particle_qd = wp.zeros_like(state.particle_q)
        interaction.track_count = interaction.track_capacity = n
        interaction._live.assign([0, n, 0])
        interaction._track_nodes = wp.array(np.arange(n * 4).reshape(n, 4), dtype=wp.vec4i, device=DEVICE)
        weights = rng.uniform(0.1, 1, (n, 4))
        weights /= weights.sum(axis=1)[:, None]
        interaction._track_bary = wp.array(weights, dtype=wp.vec4, device=DEVICE)
        binding = prepare_projection(interaction, state, 0, 0.00028)
        expected = wp.empty_like(binding.branches)
        wp.launch(
            original_material_branches,
            n,
            [
                binding.nodes,
                binding.bary,
                binding.q,
                binding.body_q,
                binding.body,
                binding.radius,
                binding.samples,
                expected,
            ],
            device=DEVICE,
        )
        np.testing.assert_array_equal(binding.branches.numpy(), expected.numpy())
        original = reference(binding)
        np.testing.assert_allclose(binding.result.numpy(), original, rtol=4e-7, atol=5e-9)

    def test_projected_force_and_pc_arithmetic_preserves_real_coupled_solution(self):
        import inspect

        from newton_solver_suturebot._src.solvers.coupled.acceleration import shaft_preconditioner
        from newton_solver_suturebot._src.solvers.coupled.interface import (
            insertion,
            shaft_projected_kernels,
            tunnel_alm,
        )

        originals = (
            insertion._shaft_forces,
            tunnel_alm._tunnel_forces,
            shaft_preconditioner.prepare_anchors,
            shaft_preconditioner.prepare_alm_anchors,
        )
        replacements = (
            shaft_projected_kernels.shaft_forces_projected,
            shaft_projected_kernels.tunnel_forces_projected,
            shaft_projected_kernels.prepare_anchors_projected,
            shaft_projected_kernels.prepare_alm_anchors_projected,
        )
        call_names = {kernel: list(inspect.signature(kernel.func).parameters) for kernel in originals}
        replaced = dict(zip(originals, replacements, strict=True))
        seen = set()
        launch = wp.launch

        def from_original_projection(kernel, dim, inputs, *args, **kwargs):
            if kernel not in replaced:
                return launch(kernel, dim, inputs, *args, **kwargs)
            values = dict(zip(call_names[kernel], inputs, strict=True))
            radius = values.get("tangent_radius", values.get("radius", 0.5 * values.get("engagement_radius", 0)))
            q = values["q"]
            packet = wp.empty((len(values["nodes"]), 8), dtype=float, device=q.device)
            launch(
                original_packet,
                len(values["nodes"]),
                [
                    values["nodes"],
                    values["bary"],
                    q,
                    values.get("body_q", values.get("poses")),
                    values["body"],
                    radius,
                    values["samples"],
                    packet,
                ],
                device=q.device,
            )
            seen.add(kernel)
            return launch(replaced[kernel], dim, [*inputs, packet], *args, **kwargs)

        for use_alm in (False, True):
            outputs = []
            for specialized in (False, True):
                with _example() as ex:
                    if use_alm:
                        c, _alm = _prepare(ex, friction=0.02)
                        h = _STEP
                    else:
                        h, _native, c = native_embedded(ex)
                    from contextlib import nullcontext

                    context = (
                        patch.object(wp, "launch", side_effect=from_original_projection)
                        if specialized
                        else nullcontext()
                    )
                    with context:
                        c.step(ex.state_0, ex.state_1, ex.control, h)
                    self.assertLess(ex.interaction.force_balance_error, 1e-8)
                    self.assertLess(ex.interaction.torque_balance_error, 1e-9)
                    self.assertLessEqual(c.iterations, c.max_iterations)
                    outputs.append(_arrays(ex.state_1))
            for field in ("particle_q", "body_q"):
                np.testing.assert_allclose(outputs[1][field], outputs[0][field], rtol=2e-6, atol=2e-9)
        from test_needle_alm_preconditioner import NeedleALMPreconditionerTests

        fixture = NeedleALMPreconditionerTests()
        fixture.setUp()
        expected = fixture.prepare(slip=1e-5)
        with patch.object(wp, "launch", side_effect=from_original_projection):
            observed = fixture.prepare(slip=1e-5)
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(seen, set(originals))

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "Captured native force/preparation")
    def test_native_force_graph_and_preconditioner_share_fresh_projection(self):

        with _example() as ex:
            h, _native, coupler = native_embedded(ex)
            ex.interaction.cooperative_shaft_projection = True
            coupler.capture_coupling = True
            coupler.capture_fem = True
            coupler.step(ex.state_0, ex.state_1, ex.control, h)
            graph = coupler._host_trial_forces
            self.assertIsNotNone(graph)
            self.assertGreater(graph.launches, 0)
            binding = ex.interaction._cooperative_shaft_binding
            self.assertIs(graph._projection_binding, binding)
            self.assertGreater(binding.counts.numpy()[0], 0)
            self.assertGreater(binding.counts.numpy()[1], 0)
            pc = coupler._shaft_preconditioner
            self.assertIsNotNone(pc._input_preparation)
            self.assertGreater(pc._input_preparation.replay.launches, 0)
            self.assertTrue(any(a is binding.result for a in pc._input_preparation.replay.owners))
            self.assertLessEqual(coupler.iterations, coupler.max_iterations)
            self.assertLess(ex.interaction.force_balance_error, 1e-8)
            self.assertLess(ex.interaction.torque_balance_error, 1e-9)

    def test_real_alm_rejection_restores_history_and_retry_matches_original(self):
        results = []
        for enabled in (False, True):
            with _example() as ex:
                coupler, alm = _prepare(ex, friction=0.02)
                ex.interaction.cooperative_shaft_projection = enabled
                if enabled:
                    before, native = _arrays(ex.state_0), _native_state(ex.rigid_solver)
                    with patch.object(
                        ex.interaction, "check_step_resolution", side_effect=RuntimeError("late rejection")
                    ):
                        with self.assertRaisesRegex(RuntimeError, "late rejection"):
                            coupler.step(ex.state_0, ex.state_1, ex.control, _STEP)
                    np.testing.assert_array_equal(_native_state(ex.rigid_solver), native)
                    for name, value in before.items():
                        np.testing.assert_array_equal(getattr(ex.state_0, name).numpy(), value)
                    np.testing.assert_array_equal(alm.committed_normal.numpy(), 0)
                    self.assertFalse(alm._open)
                coupler.step(ex.state_0, ex.state_1, ex.control, _STEP)
                self.assertTrue(alm.converged(coupler.absolute_tolerance, coupler.relative_tolerance))
                self.assertLess(ex.interaction.force_balance_error, 1e-8)
                self.assertLess(ex.interaction.torque_balance_error, 1e-9)
                results.append((_arrays(ex.state_1), alm.committed_normal.numpy().copy()))
                if enabled and ex.device.is_cuda:
                    self.assertGreater(ex.interaction._cooperative_shaft_binding.counts.numpy()[0], 0)
        for field in ("particle_q", "body_q"):
            np.testing.assert_allclose(results[1][0][field], results[0][0][field], rtol=2e-6, atol=2e-9)
        # Actual physical coupled acceptance remains the original certificate;
        # rounded geometry may follow different convergent numerical iterates.


if __name__ == "__main__":
    unittest.main()
