"""Ordered single-trial facet reuse, full overflow recovery and physical parity."""

import ast
import copy
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import test_surface_contact_tangent as fixture_module
import warp as wp
from test_surface_contact_tangent import DEVICE, block, evaluate, fixture

from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_contact_tangent import SurfaceTangentAnchors
from newton_solver_suturebot._src.solvers.coupled.interface import surface_facet_scratch as module
from newton_solver_suturebot._src.solvers.coupled.interface.surface_contact_forces import surface_forces_with_tangent


def recorded(anchors):
    count = int(anchors.count.numpy()[0])
    n = min(count, anchors.matrices.shape[0])
    sources = anchors.sources.numpy()[:n]
    order = np.lexsort((sources[:, 1], sources[:, 0])) if n else np.array([], dtype=int)
    return [anchors.count.numpy(), anchors.flags.numpy()] + [
        getattr(anchors, name).numpy()[:n][order]
        for name in ("nodes", "bary", "points", "levers", "normals", "tangents", "matrices", "sources", "scalars")
    ]


def evaluate_cached(data, scratch, anchors, *, implicit=True):
    """Eager component oracle on CPU too; production helper remains CUDA capture only."""
    original = wp.launch
    calls = []

    def launch(kernel, *args, **kwargs):
        if kernel is surface_forces_with_tangent:
            arguments = args[1]
            scratch.collect(arguments)
            overflow = bool(scratch.overflow.numpy()[0])
            calls.append(overflow)
            if not overflow:
                scratch.force(arguments)
                return
        return original(kernel, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=launch):
        result = evaluate(data, anchors, implicit=implicit)
    if len(calls) != 1:
        raise AssertionError(f"Expected one actual recorded surface call, got {calls}")
    return result, calls[0]


def arguments_for(data, anchors, *, implicit=True):
    arguments = []
    original = wp.launch

    def launch(kernel, *args, **kwargs):
        if kernel is surface_forces_with_tangent:
            arguments.append(list(args[1]))
        return original(kernel, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=launch):
        evaluate(data, anchors, implicit=implicit)
    if len(arguments) != 1:
        raise AssertionError("The exact surface fixture did not launch once")
    return arguments[0]


def clear_outputs(arguments):
    # Original particle/body/diagnostics/contact flags + recording count/flags.
    for index in (31, 32, 33, 34, 36, 37):
        arguments[index].zero_()


def outputs(arguments):
    return tuple(arguments[index].numpy().copy() for index in (32, 31, 33, 34))


class SurfaceFacetScratchTests(unittest.TestCase):
    assert_physical_equal = fixture_module.SurfaceTangentTests.assert_physical_equal

    def setUp(self):
        wp.init()

    def compare(self, data, *, capacity=64, implicit=True, expect_overflow=False):
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        expected = evaluate(data, anchors, implicit=implicit)
        expected_recording = recorded(anchors)
        scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE, capacity=capacity)
        actual, overflow = evaluate_cached(data, scratch, anchors, implicit=implicit)
        self.assertEqual(overflow, expect_overflow)
        self.assert_physical_equal(expected, actual, data=data)
        for a, b in zip(expected_recording, recorded(anchors), strict=True):
            np.testing.assert_array_equal(a, b)
        return scratch, actual

    def test_single_sample_multifacet_force_and_recording_exact(self):
        data = fixture(single_active_sample=True)
        scratch, result = self.compare(data)
        self.assertGreater(int(scratch.counts.numpy()[0]), 1)
        self.assertEqual(int(scratch.counts.numpy()[1]), -1)
        self.assertGreater(float(np.linalg.norm(result[0])), 0)

    def test_multisample_fixed_and_unmapped_reactions_and_impulse_limit(self):
        for implicit in (False, True):
            with self.subTest(implicit=implicit):
                self.compare(fixture(fixed=[1], unmapped=[3]), implicit=implicit)

    def test_unpenetrated_facets_remain_in_normalization(self):
        first, tets = block(slope=0, thickness=0.00005)
        second, other = block(slope=0, thickness=0.00005)
        second[:, 2] -= 0.00015
        data = fixture(np.vstack((first, second)), np.vstack((tets, other + len(first))))
        self.compare(data)

    def test_overflow_uses_whole_original_without_partial_forces(self):
        for capacity in (0, 1):
            with self.subTest(capacity=capacity):
                self.compare(fixture(), capacity=capacity, expect_overflow=True)

    def test_recollection_replaces_geometry_permissions_and_no_hit(self):
        data = fixture(single_active_sample=True)
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE)
        for z in (0.0004, 0.0003, 0.1, 0.0004):
            pose = data[1].body_q.numpy()
            pose[0, 2] = z
            data[1].body_q.assign(pose)
            expected = evaluate(data, anchors)
            expected_recording = recorded(anchors)
            scratch.permissions.fill_(1)  # Poison old permissions: collect must replace every used entry.
            actual, overflow = evaluate_cached(data, scratch, anchors)
            self.assertFalse(overflow)
            self.assert_physical_equal(expected, actual)
            for a, b in zip(expected_recording, recorded(anchors), strict=True):
                np.testing.assert_array_equal(a, b)
            if z == 0.1:
                np.testing.assert_array_equal(scratch.counts.numpy(), -1)
                np.testing.assert_array_equal(actual[0], 0)

    def test_permission_and_opening_changes_are_read_for_every_collection(self):
        data = fixture(single_active_sample=True)
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        arguments = arguments_for(data, anchors)
        scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE)
        for allowed in (4, 0, 4):
            arguments[26].fill_(allowed)
            clear_outputs(arguments)
            wp.launch(surface_forces_with_tangent, scratch.samples, arguments, device=DEVICE, block_dim=1)
            expected, records = outputs(arguments), recorded(anchors)
            clear_outputs(arguments)
            scratch.collect(arguments)
            self.assertEqual(int(scratch.overflow.numpy()[0]), 0)
            scratch.force(arguments)
            self.assert_physical_equal(expected, outputs(arguments))
            for a, b in zip(records, recorded(anchors), strict=True):
                np.testing.assert_array_equal(a, b)

    def test_near_gap_radius_and_permission_boundaries(self):
        data = fixture(single_active_sample=True)
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE)
        # The flat top is exactly z=0; approach both sides of zero and the
        # physical radius by one float32 neighbor, without relaxing branches.
        vertices, tets = block(slope=0)
        data = fixture(vertices, tets, single_active_sample=True)
        arguments = arguments_for(data, anchors)
        for boundary in (np.float32(0), np.float32(0.0005)):
            for z in (np.nextafter(boundary, np.float32(-1)), boundary, np.nextafter(boundary, np.float32(1))):
                pose = data[1].body_q.numpy()
                pose[0, 2] = z
                data[1].body_q.assign(pose)
                for allowed in (0, 4):
                    arguments[26].fill_(allowed)
                    clear_outputs(arguments)
                    wp.launch(surface_forces_with_tangent, scratch.samples, arguments, device=DEVICE, block_dim=1)
                    expected, records = outputs(arguments), recorded(anchors)
                    clear_outputs(arguments)
                    scratch.collect(arguments)
                    self.assertEqual(int(scratch.overflow.numpy()[0]), 0)
                    scratch.force(arguments)
                    self.assert_physical_equal(expected, outputs(arguments))
                    for a, b in zip(records, recorded(anchors), strict=True):
                        np.testing.assert_array_equal(a, b)

    def test_public_fast_path_declines_eager_cpu_custom_and_gradient_calls(self):
        data = fixture()
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        arguments = arguments_for(data, anchors)
        scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE)
        self.assertFalse(scratch.launch(surface_forces_with_tangent, arguments))
        with patch.object(scratch, "collect", side_effect=AssertionError("Unsupported call collected")):
            self.assertFalse(scratch.launch(object(), arguments))
            with wp.Tape():
                self.assertFalse(scratch.launch(surface_forces_with_tangent, arguments))

    def test_original_pass_arithmetic_and_lazy_permission_arguments_are_unchanged(self):
        original = ast.parse(inspect.getsource(surface_forces_with_tangent.func)).body[0]
        cached = ast.parse(inspect.getsource(module.forces_from_facets.func)).body[0]
        original_loops = [n for n in original.body if isinstance(n, ast.While)]
        cached_loops = [n for n in cached.body if isinstance(n, ast.For)]
        self.assertEqual(len(original_loops), 3)
        self.assertEqual(len(cached_loops), 3)

        class Normalize(ast.NodeTransformer):
            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "_surface_facet":
                    return None
                if isinstance(node.value, ast.Subscript) and ast.unparse(node.value.value).startswith(
                    ("face_ids", "saved_")
                ):
                    return None
                # Nested vec3 component indexing (saved_geometry[i, facet_slot][0]).
                if isinstance(node.value, ast.Subscript) and ast.unparse(node.value).startswith("saved_"):
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                if ast.unparse(node.test) == "permissions[i, facet_slot] < 0":
                    self.assert_lazy(node)
                    return copy.deepcopy(node.body[0])
                return self.generic_visit(node)

            @staticmethod
            def assert_lazy(node):
                assert len(node.body) == 2 and len(node.orelse) == 1
                assert ast.unparse(node.body[1]) == "permissions[i, facet_slot] = int(permitted)"
                assert ast.unparse(node.orelse[0]) == "permitted = permissions[i, facet_slot] != 0"

            def visit_Attribute(self, node):
                if ast.unparse(node) == "hit.sign":
                    return ast.parse("sides[i]", mode="eval").body
                return self.generic_visit(node)

        for expected, actual in zip(original_loops, cached_loops, strict=True):
            a = Normalize().visit(ast.Module(body=copy.deepcopy(expected.body), type_ignores=[]))
            b = Normalize().visit(ast.Module(body=copy.deepcopy(actual.body), type_ignores=[]))
            self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False))
        # Nearest/sign/local-passage and AABB extents precede collection unchanged.
        collector = ast.parse(inspect.getsource(module.collect_facets.func)).body[0]
        before_min = next(i for i, n in enumerate(original.body) if ast.unparse(n).startswith("minimum ="))
        collector_prefix = [n for n in collector.body if ast.unparse(n) != "counts[i] = -1"][:before_min]
        self.assertEqual(
            ast.dump(ast.Module(body=original.body[:before_min], type_ignores=[]), include_attributes=False),
            ast.dump(ast.Module(body=collector_prefix, type_ignores=[]), include_attributes=False),
        )

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "conditional CUDA graph")
    def test_real_host_graph_owns_scratch_and_rebinds_without_stale_storage(self):
        from test_surface_pc_capture import contact, prepare_forces

        with contact(capture=True) as (ex, c, _native, pc):
            graph = prepare_forces(ex, c, pc)
            scratch = graph.permissions.surface_scratch
            self.assertIsNotNone(scratch)
            graph.evaluate(ex.sim_dt, False)
            self.assertGreater(graph.launches, 0)
            self.assertGreater(int(scratch.counts.numpy().max()), 0)
            self.assertEqual(int(scratch.overflow.numpy()[0]), 0)
            first_signature = graph.signature
            first_graphs = tuple(graph.graphs.values())
            old_counts = scratch.counts
            scratch.counts = wp.empty_like(old_counts)
            graph.prepare(ex.sim_dt)
            self.assertNotEqual(graph.signature, first_signature)
            self.assertFalse(graph.graphs)
            graph.evaluate(ex.sim_dt, False)
            self.assertGreater(int(scratch.counts.numpy().max()), 0)
            self.assertTrue(first_graphs)  # Retained owner during rebinding check.
            np.testing.assert_array_equal(old_counts.numpy(), scratch.counts.numpy())
            before = scratch.counts.numpy().copy()
            graph.evaluate(ex.sim_dt, False, fresh=False)
            np.testing.assert_array_equal(scratch.counts.numpy(), before)
            self.assertGreater(graph.primal_reuses, 0)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "conditional CUDA graph")
    def test_unrecorded_kernel_uses_same_scratch_without_recorder_writes(self):
        data = fixture(single_active_sample=True)
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        arguments = arguments_for(data, anchors)
        physical = arguments[:35]
        scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE)
        with wp.ScopedCapture(device=DEVICE) as capture:
            self.assertTrue(scratch.launch(module._ORIGINAL_PHYSICAL_KERNEL, physical))
        clear_outputs(arguments)
        wp.launch(module._ORIGINAL_PHYSICAL_KERNEL, scratch.samples, physical, device=DEVICE, block_dim=1)
        expected = outputs(arguments)
        clear_outputs(arguments)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(scratch.overflow.numpy()[0]), 0)
        self.assert_physical_equal(expected, outputs(arguments))
        np.testing.assert_array_equal(scratch.disabled_recording.flags.numpy(), 0)
        self.assertEqual(scratch.disabled_recording.count.size, 0)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda, "conditional CUDA graph")
    def test_actual_conditional_graph_selects_cached_or_original_and_replays_current_inputs(self):
        data = fixture(single_active_sample=True)
        anchors = SurfaceTangentAnchors(128, device=DEVICE)
        arguments = arguments_for(data, anchors)
        for capacity in (0, 64):
            scratch = module.SurfaceFacetScratch(len(data[0].samples), DEVICE, capacity=capacity)
            with wp.ScopedCapture(device=DEVICE) as capture:
                self.assertTrue(scratch.launch(surface_forces_with_tangent, arguments))
            for allowed in (0, 4, 0):
                arguments[26].fill_(allowed)
                clear_outputs(arguments)
                wp.launch(surface_forces_with_tangent, scratch.samples, arguments, device=DEVICE, block_dim=1)
                expected, records = outputs(arguments), recorded(anchors)
                clear_outputs(arguments)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(scratch.overflow.numpy()[0]), int(capacity == 0))
                self.assert_physical_equal(expected, outputs(arguments))
                for a, b in zip(records, recorded(anchors), strict=True):
                    np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
