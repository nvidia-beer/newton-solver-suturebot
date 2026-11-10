# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Static permission oracle checks; no new physical integration or prescribed outcome."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import setup

from newton_solver_suturebot._src.geometry.needle_queries import NeedleQueryGeometry
from newton_solver_suturebot._src.insertion import _surface_crossings, _surface_positions
from newton_solver_suturebot._src.needle_trial_forces import _NeedleTrialForces, static_crossing
from newton_solver_suturebot._src.tunnel_alm import _TunnelALM


def prepare(x, state, body):
    q = state.particle_q.numpy().astype(float)
    pose = state.body_q.numpy()[body]
    samples = x._body_samples(state, body, pose)
    tip = x._body_tip(state, body, pose)
    wp.launch(_surface_positions, len(x.mapping), [state.particle_q, x._mapping, x._surface], device=x.device)
    x.mesh.refit()
    wp.copy(x._active_surface, state.particle_q, count=len(x.vertices))
    x._active_mesh.refit()
    if x._device_queries is None:
        x._device_queries = NeedleQueryGeometry(x.tetrahedra, len(x.vertices), device=x.device)
    if x._previous_q is not None:
        x._device_queries.commit(wp.array(x._previous_q, dtype=wp.vec3, device=x.device))
    x._prepare_device_queries(state, q, np.vstack((tip, samples)))
    return q, samples


@wp.kernel
def crossing_probe(
    starts: wp.array[wp.vec3d],
    ends: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    faces: wp.array[wp.vec3i],
    mesh: wp.uint64,
    out: wp.array[int],
):
    i = wp.tid()
    out[i] = int(static_crossing(starts[i], ends[i], q, faces, mesh))


class NeedleTrialForcesTests(unittest.TestCase):
    def test_deformed_rotated_shaft_masks_and_exact_force_outputs(self):
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            rng = np.random.default_rng(1901)
            initial_pose = ex.state_0.body_q.numpy().copy()
            initial_q = ex.state_0.particle_q.numpy().copy()
            for _trial in range(8):
                pose = initial_pose.copy()
                pose[0, :3] += rng.normal(size=3) * 0.00007
                q = initial_q.copy()
                q[ex.fixed_vertex_indices] = initial_q[ex.fixed_vertex_indices]
                free = np.ones(len(q), bool)
                free[ex.fixed_vertex_indices] = False
                q[free] += rng.normal(size=(free.sum(), 3)) * 0.00001
                ex.state_0.particle_q.assign(q)
                ex.state_0.body_q.assign(pose)
                q, samples = prepare(x, ex.state_0, ex.needle_body)
                expected = x._visited_sample_mask(ex.state_0, ex.needle_body, q, samples=samples)
                device = _NeedleTrialForces(x)
                device.evaluate(ex.state_0, ex.needle_body)
                actual = device.mask.numpy()
                fallback = int(device.fallback.numpy()[0])
                np.testing.assert_array_equal(actual[1:], expected[1:])
                if not fallback:
                    np.testing.assert_array_equal(actual, expected)
                # Only the conservative tip certificate may choose CPU fallback.
                # Compare the shaft-derived forces with the same exact tip oracle.
                actual[0] = expected[0]
                outputs = []
                for mask in (expected, actual):
                    ex.state_0.body_f.zero_()
                    ex._external_forces.zero_()
                    with patch.object(x, "_visited_sample_mask", return_value=mask):
                        x.add_forces(
                            ex.state_0,
                            ex.needle_body,
                            ex.drive.com,
                            ex._external_forces,
                            0.0005,
                            trial=True,
                            implicit=True,
                        )
                    outputs.append(
                        (
                            ex._external_forces.numpy().copy(),
                            ex.state_0.body_f.numpy().copy(),
                            x._contact_flags.numpy().copy(),
                        )
                    )
                for a, b in zip(*outputs, strict=True):
                    np.testing.assert_array_equal(a, b)

    def test_prepare_storage_and_complete_penalty_alm_force_parity(self):
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            prepare(x, ex.state_0, ex.needle_body)
            device = _NeedleTrialForces(x)
            arrays = (
                "visited",
                "nodes",
                "bary",
                "points",
                "segments",
                "projected_arc",
                "associated",
                "opening_nodes",
                "opening_bary",
                "old_tip",
                "step_flags",
                "mask",
                "fallback",
            )
            pointers = {name: getattr(device, name).ptr for name in arrays}
            signature = device.signature()
            self.assertFalse(device.prepare(x))
            self.assertEqual(signature, device.signature())
            self.assertEqual(pointers, {name: getattr(device, name).ptr for name in arrays})
            np.testing.assert_array_equal(device.old_tip.numpy()[0], x._previous_tip)
            saved = (x.track_count, x.puncture_count, x.entry_count, x.exit_count, tuple(x._track_edges))
            for alm in (False, True):
                if alm:
                    x.tunnel_alm = _TunnelALM(
                        x, penalty_per_m=60000, compliance_per_m=1 / 60000, clearance=0.00002, tolerance=0.000002
                    )
                    x.tunnel_alm.begin_step()
                outputs = []
                for original in (True, False):
                    ex.state_0.body_f.zero_()
                    ex._external_forces.zero_()
                    if original:
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
                        device.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, 0.0005)
                    outputs.append(
                        [
                            ex._external_forces.numpy().copy(),
                            ex.state_0.body_f.numpy().copy(),
                            x._contact_flags.numpy().copy(),
                            x._diagnostics.numpy().copy(),
                        ]
                    )
                    if alm:
                        outputs[-1].extend(
                            [
                                x.tunnel_alm.proposed_normal.numpy().copy(),
                                x.tunnel_alm.proposed_tangent.numpy().copy(),
                                x.tunnel_alm._statistics.numpy().copy(),
                            ]
                        )
                self.assertEqual(int(device.fallback.numpy()[0]), 0)
                for a, b in zip(*outputs, strict=True):
                    np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-10)
                self.assertEqual(
                    saved, (x.track_count, x.puncture_count, x.entry_count, x.exit_count, tuple(x._track_edges))
                )
                if alm:
                    np.testing.assert_array_equal(x.tunnel_alm.committed_normal.numpy(), 0)
                    x.tunnel_alm.discard()
            # Scalar settings and replaced force buffers must invalidate capture,
            # while accepted metadata contents are updated without new storage.
            signature = device.signature()
            x.friction += 0.01
            self.assertNotEqual(signature, device.signature())
            signature = device.signature()
            x._track_lengths = wp.clone(x._track_lengths)
            self.assertNotEqual(signature, device.signature())
            signature = device.signature()
            # Accepted metadata is refreshed in place, never captured as stale scalars.
            x._previous_tip = x._previous_tip + np.array([1e-8, 0, 0])
            self.assertFalse(device.prepare(x))
            np.testing.assert_array_equal(device.old_tip.numpy()[0], x._previous_tip)
            self.assertEqual(signature, device.signature())
            self.assertEqual(pointers, {name: getattr(device, name).ptr for name in arrays})

    def test_cuda_capture_replays_updated_trial_geometry(self):
        if not wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda:
            self.skipTest("CUDA graph capture")
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            prepare(x, ex.state_0, ex.needle_body)
            device = _NeedleTrialForces(x)
            device.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, 0.0005)
            with wp.ScopedCapture(device=x.device) as capture:
                ex.state_0.body_f.zero_()
                ex._external_forces.zero_()
                device.add_forces(ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, 0.0005)
            pose = ex.state_0.body_q.numpy().copy()
            for dz in (0.0, 0.00001, -0.00001):
                moved = pose.copy()
                moved[0, 2] += dz
                ex.state_0.body_q.assign(moved)
                wp.capture_launch(capture.graph)
                actual = ex._external_forces.numpy().copy()
                self.assertEqual(int(device.fallback.numpy()[0]), 0)
                ex.state_0.body_f.zero_()
                ex._external_forces.zero_()
                x.add_forces(
                    ex.state_0, ex.needle_body, ex.drive.com, ex._external_forces, 0.0005, trial=True, implicit=True
                )
                np.testing.assert_allclose(actual, ex._external_forces.numpy(), rtol=2e-6, atol=1e-10)

    def test_static_crossings_match_float64_cpu_endpoints_and_clear_paths(self):
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            q, _ = prepare(x, ex.state_0, ex.needle_body)
            rng = np.random.default_rng(1902)
            starts = rng.uniform([-0.008, -0.003, -0.002], [0.008, 0.003, 0.002], (150, 3))
            ends = rng.uniform([-0.008, -0.003, -0.002], [0.008, 0.003, 0.002], (150, 3))
            # Include stationary and exact surface endpoint cases.
            starts = np.vstack((starts, q, q))
            ends = np.vstack((ends, q, q + [0, 0, 0.0001]))
            expected = [
                bool(_surface_crossings(a, b, q, q, x._active_triangles)) for a, b in zip(starts, ends, strict=True)
            ]
            result = wp.empty(len(starts), dtype=int, device=x.device)
            wp.launch(
                crossing_probe,
                len(starts),
                [
                    wp.array(starts, dtype=wp.vec3d, device=x.device),
                    wp.array(ends, dtype=wp.vec3d, device=x.device),
                    ex.state_0.particle_q,
                    wp.array(x._active_triangles, dtype=wp.vec3i, device=x.device),
                    x._active_mesh.id,
                    result,
                ],
                device=x.device,
            )
            np.testing.assert_array_equal(result.numpy(), expected)

    def test_boundary_motion_requests_fallback_and_does_not_commit_history(self):
        with _example() as ex:
            setup(ex, 0.0005, 1500)
            x = ex.interaction
            saved = (x.track_count, x.puncture_count, x.entry_count, x.exit_count, tuple(x._track_edges))
            pose = ex.state_0.body_q.numpy().copy()
            pose[0, 2] += 0.003
            ex.state_0.body_q.assign(pose)
            prepare(x, ex.state_0, ex.needle_body)
            device = _NeedleTrialForces(x)
            device.evaluate(ex.state_0, ex.needle_body)
            self.assertEqual(int(device.fallback.numpy()[0]), 1)
            self.assertEqual(
                saved, (x.track_count, x.puncture_count, x.entry_count, x.exit_count, tuple(x._track_edges))
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
