"""GPU shaft permission keeps exact CPU tip rules and force/history behavior."""

import gc
import os
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import setup
from test_needle_trial_forces import prepare

from newton_solver_suturebot._src.tunnel_alm import _TunnelALM


class HybridSampleMaskTests(unittest.TestCase):
    def require_cuda(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("CUDA shaft association")

    def test_complete_masks_and_tip_exit_points_match_cpu_under_deformation(self):
        self.require_cuda()
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            rng = np.random.default_rng(278)
            pose0, q0 = ex.state_0.body_q.numpy().copy(), ex.state_0.particle_q.numpy().copy()
            helper = None
            for trial in range(8):
                pose, q = pose0.copy(), q0.copy()
                pose[0, :3] += rng.normal(size=3) * 7e-5
                q += rng.normal(size=q.shape) * 1e-5
                q[ex.fixed_vertex_indices] = q0[ex.fixed_vertex_indices]
                ex.state_0.particle_q.assign(q)
                ex.state_0.body_q.assign(pose)
                q, samples = prepare(x, ex.state_0, ex.needle_body)
                x._tip_embedding = x._locate(samples[0], q)
                expected = x._visited_sample_mask(ex.state_0, ex.needle_body, q, samples=samples)
                tip_exit = x._tip_exit_point.copy()
                actual = x._device_visited_sample_mask(ex.state_0, ex.needle_body, q, samples)
                self.assertIsNotNone(actual, trial)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(x._tip_exit_point, tip_exit)
                if helper is not None:
                    self.assertIs(helper, x._host_sample_permissions)
                helper = x._host_sample_permissions

    def test_penalty_and_alm_force_fields_and_history_match_host_oracle(self):
        self.require_cuda()
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            before = (x.track_count, x.entry_count, x.exit_count, tuple(x._track_edges), x._in_material)
            for alm in (False, True):
                if alm:
                    x.tunnel_alm = _TunnelALM(
                        x,
                        penalty_per_m=60000,
                        compliance_per_m=1 / 60000,
                        clearance=0.00002,
                        tolerance=0.000002,
                    )
                    x.tunnel_alm.begin_step()
                results = []
                for host in (True, False):
                    ex.state_0.body_f.zero_()
                    ex._external_forces.zero_()
                    if host:
                        with patch.object(x, "_device_visited_sample_mask", return_value=None):
                            x.add_forces(
                                ex.state_0,
                                ex.needle_body,
                                ex.drive.com,
                                ex._external_forces,
                                0.0005,
                                trial=True,
                                implicit=True,
                            )
                    else:
                        x.add_forces(
                            ex.state_0,
                            ex.needle_body,
                            ex.drive.com,
                            ex._external_forces,
                            0.0005,
                            trial=True,
                            implicit=True,
                        )
                    arrays = [
                        ex._external_forces,
                        ex.state_0.body_f,
                        x._contact_flags,
                        x._diagnostics,
                        x._sample_in_visited_material,
                    ]
                    if alm:
                        arrays += [
                            x.tunnel_alm.proposed_normal,
                            x.tunnel_alm.proposed_tangent,
                            x.tunnel_alm._statistics,
                        ]
                    results.append([array.numpy().copy() for array in arrays])
                self.assertIsNotNone(x._host_sample_permissions)
                for a, b in zip(*results, strict=True):
                    np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-10)
                self.assertEqual(
                    before, (x.track_count, x.entry_count, x.exit_count, tuple(x._track_edges), x._in_material)
                )
                if alm:
                    np.testing.assert_array_equal(x.tunnel_alm.committed_normal.numpy(), 0)
                    x.tunnel_alm.discard()

    def test_custom_tape_grad_and_stale_geometry_use_cpu_and_replaced_topology_rebuilds(self):
        self.require_cuda()
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            q, samples = prepare(x, ex.state_0, ex.needle_body)
            x.samples = x.samples.copy()
            args = (ex.state_0, ex.needle_body, q, samples)
            self.assertIsNotNone(x._device_visited_sample_mask(*args))
            helper = x._host_sample_permissions
            with patch.object(x, "_visited_sample_mask", return_value=np.zeros(len(x.samples), dtype=np.int32)):
                self.assertIsNone(x._device_visited_sample_mask(*args))
            with wp.Tape():
                self.assertIsNone(x._device_visited_sample_mask(*args))
            original = ex.state_0.body_q
            ex.state_0.body_q = wp.clone(original, requires_grad=True)
            self.assertIsNone(x._device_visited_sample_mask(*args))
            ex.state_0.body_q = original
            self.assertIsNone(x._device_visited_sample_mask(ex.state_0, ex.needle_body, q + 1e-4, samples))
            old_owner = weakref.ref(x.samples)
            x.samples = x.samples.copy()
            gc.collect()
            self.assertIsNotNone(old_owner())
            self.assertIs(x._host_sample_owners[0], old_owner())
            self.assertIsNotNone(x._device_visited_sample_mask(*args))
            self.assertIsNot(helper, x._host_sample_permissions)
            gc.collect()
            self.assertIsNone(old_owner())
            helper = x._host_sample_permissions
            x._fixed = wp.clone(x._fixed)
            self.assertIsNotNone(x._device_visited_sample_mask(*args))
            self.assertIsNot(helper, x._host_sample_permissions)

    def test_degenerate_visited_tetrahedron_requests_original_cpu_path(self):
        self.require_cuda()
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            q = ex.state_0.particle_q.numpy().copy()
            ids = x.tetrahedra[next(iter(x._visited_tets))]
            q[ids[3]] = q[ids[2]]
            pose = ex.state_0.body_q.numpy().copy()
            tip = x._body_tip(ex.state_0, ex.needle_body, pose[ex.needle_body])
            pose[ex.needle_body, :3] += q[ids[:3]].mean(axis=0) - tip
            ex.state_0.particle_q.assign(q)
            ex.state_0.body_q.assign(pose)
            q, samples = prepare(x, ex.state_0, ex.needle_body)
            helper = x._host_sample_permissions
            with patch.object(helper, "evaluate", wraps=helper.evaluate) as evaluate:
                self.assertIsNone(x._device_visited_sample_mask(ex.state_0, ex.needle_body, q, samples))
                evaluate.assert_not_called()
            self.assertIs(x._host_sample_permissions, helper)
            with self.assertRaises(np.linalg.LinAlgError):
                x._visited_sample_mask(ex.state_0, ex.needle_body, q, samples=samples)

    def test_degenerate_visited_cell_away_from_samples_preserves_cpu_failure(self):
        self.require_cuda()
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            q = ex.state_0.particle_q.numpy().copy()
            ids = x.tetrahedra[next(iter(x._visited_tets))]
            q[ids] += 10
            q[ids[3]] = q[ids[2]]
            ex.state_0.particle_q.assign(q)
            q, samples = prepare(x, ex.state_0, ex.needle_body)
            self.assertGreater(np.linalg.norm(samples[:, None] - q[ids], axis=2).min(), 1)
            helper = x._host_sample_permissions
            with patch.object(helper, "evaluate", wraps=helper.evaluate) as evaluate:
                self.assertIsNone(x._device_visited_sample_mask(ex.state_0, ex.needle_body, q, samples))
                evaluate.assert_not_called()
            self.assertIs(x._host_sample_permissions, helper)
            with self.assertRaises(np.linalg.LinAlgError):
                x._visited_sample_mask(ex.state_0, ex.needle_body, q, samples=samples)

    def test_empty_history_skips_helper_allocation(self):
        with _example() as ex:
            ex.select_regions([1])
            ex.select_point_xy("entry", -0.004, 0)
            ex.select_point_xy("exit", 0.004, 0)
            ex.preview()
            ex.start()
            ex.running = False
            x = ex.interaction
            q = ex.state_0.particle_q.numpy().astype(float)
            samples = x._body_samples(ex.state_0, ex.needle_body)
            self.assertIsNone(x._device_visited_sample_mask(ex.state_0, ex.needle_body, q, samples))
            self.assertIsNone(x._host_sample_permissions)


if __name__ == "__main__":
    unittest.main()
