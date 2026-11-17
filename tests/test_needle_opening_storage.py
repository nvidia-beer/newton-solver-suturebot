# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The registered openings live in storage that the interaction makes once and writes in place.

The storage has ``opening_capacity`` rows; the kernels read the rows below the live count on the
device. A row from the live count on holds zeros: nodes 0 and weights 0, which is a point at the
world origin. These tests put the needle tip next to the origin, where such a row would matter if
a kernel read it, and compare with storage of exactly the live size. A reserve that is too small
doubles, is counted, and gives the results of a reserve that was large enough.
"""

import os
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_OPENINGS
from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import (
    FALLBACK_REASONS,
    _NeedleTrialForces,
)

DEVICE = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
VERTICES = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float32)
DT = 0.0005
STORAGE = ("_all_opening_faces", "_opening_points", "_opening_nodes", "_opening_bary_d", "_visited", "_live")
REUSE = FALLBACK_REASONS.index("tip within reach of a deformed opening")


def build(position, **options):
    """One tetrahedron with a corner at the origin and a needle below its face z = 0, tip first."""
    tets = np.array([[0, 1, 2, 3]], np.int32)
    samples = np.zeros((11, 3))
    samples[:, 2] = np.linspace(0, -0.01, 11)
    interaction = NeedleTissueInteraction(
        VERTICES, tets, boundary_faces(tets)[0], (), samples, 0.0005, samples[0], device=DEVICE, **options
    )
    state = SimpleNamespace(
        particle_q=wp.array(VERTICES, dtype=wp.vec3, device=DEVICE),
        particle_qd=wp.zeros(4, dtype=wp.vec3, device=DEVICE),
        body_q=wp.array([[*position, 0, 0, 0, 1]], dtype=wp.transform, device=DEVICE),
        body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE),
        body_f=wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE),
    )
    return interaction, state, wp.zeros(4, dtype=wp.vec3, device=DEVICE)


def bottom_face(interaction):
    """The surface face in the plane z = 0, which the needle meets."""
    return next(i for i, ids in enumerate(interaction.surface_triangles) if set(ids) == {0, 1, 2})


def host_trial(interaction, state, forces):
    """The host force evaluation of a candidate, as a coupling trial makes it."""
    state.body_f.zero_()
    forces.zero_()
    interaction.add_forces(state, 0, np.zeros(3), forces, DT, trial=True, implicit=True)
    return {
        "particle_f": forces.numpy().copy(),
        "body_f": state.body_f.numpy().copy(),
        "diagnostics": interaction._diagnostics.numpy().copy(),
        "contact_flags": interaction._contact_flags.numpy().copy(),
        "sample_mask": interaction._sample_in_visited_material.numpy().copy(),
    }


def device_trial(interaction, helper, state, forces, *, skip=()):
    """The device force evaluation of a candidate; ``skip`` names kernels that are not launched."""
    helper.prepare(interaction)
    state.body_f.zero_()
    forces.zero_()
    # The current positions of the openings are scratch that the helper writes for itself.
    interaction._opening_points.zero_()
    launch = wp.launch

    def launch_but(kernel, *arguments, **options):
        return None if kernel.key in skip else launch(kernel, *arguments, **options)

    with patch.object(wp, "launch", side_effect=launch_but):
        helper.add_forces(state, 0, np.zeros(3), forces, DT)
    # The helper refitted the device geometry: the host path must not trust its cached positions.
    interaction._device_query_positions = None
    return {
        "particle_f": forces.numpy().copy(),
        "body_f": state.body_f.numpy().copy(),
        "diagnostics": interaction._diagnostics.numpy().copy(),
        "contact_flags": interaction._contact_flags.numpy().copy(),
        "mask": helper.mask.numpy().copy(),
        "fallback": helper.fallback.numpy().copy(),
    }


def accept(interaction, state, forces, position):
    """One accepted implicit step to ``position``: trial, resolution check, retained trial, acceptance."""
    start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
    state.body_q.assign([[*position, 0, 0, 0, 1]])
    trial = host_trial(interaction, state, forces)
    interaction.check_step_resolution(start, state, 0)
    interaction.retain_trial_forces(start, 0)
    interaction.accept(state, 0)
    if interaction.status == "failed":
        raise AssertionError(interaction.reason)
    return trial


def three_punctures():
    """Tip positions [m] that puncture, enter and leave at (2, 2) mm, then at (5, 2) mm, then enter at (2, 5) mm.

    Every step is shorter than the step resolution of this needle (0.303 mm).
    """
    enter, leave = np.linspace(-0.00105, 0.0012, 16), np.linspace(0.00105, -0.0012, 16)
    path = [(0.002, 0.002, z) for z in np.r_[-0.0001, np.linspace(0, 0.0012, 9), leave]]
    path += [(x, 0.002, -0.0012) for x in np.linspace(0.002, 0.005, 13)[1:]]
    path += [(0.005, 0.002, z) for z in np.r_[enter, leave]]
    path += [(0.005 - s, 0.002 + s, -0.0012) for s in np.linspace(0, 0.003, 18)[1:]]
    path += [(0.002, 0.005, z) for z in enter]
    return path


class OpeningStorageTests(unittest.TestCase):
    def assert_same(self, actual, expected, label):
        """Bit for bit on the CPU. On CUDA sums over needle samples have no fixed order: integers stay exact."""
        self.assertEqual(actual.keys(), expected.keys(), label)
        for name, value in actual.items():
            if DEVICE.is_cuda and value.dtype.kind == "f":
                np.testing.assert_allclose(value, expected[name], rtol=2e-6, atol=1e-10, err_msg=f"{label}: {name}")
            else:
                self.assertEqual(value.tobytes(), expected[name].tobytes(), f"{label}: {name}")

    def near_origin(self, openings, **options):
        """A needle whose tip was accepted 0.66 mm from the origin and is now tried 0.49 mm from it.

        ``openings`` far openings are registered first. No contact here authorizes a puncture, so
        the only history is what the test registers.
        """
        x, state, forces = build((0.0002, 0.0002, -0.0006), puncture_force=1.0, **options)
        face = bottom_face(x)
        ids = x.surface_triangles[face]
        for index in range(openings):
            # (8, 1, 0) mm and (1, 8, 0) mm: both further from the tip than any capture radius.
            near = 2 - index
            x._register_opening(ids, np.asarray([0.8 if v == near else 0.1 for v in ids]), face)
        accept(x, state, forces, (0.0002, 0.0002, -0.0006))
        state.body_q.assign([[0.0002, 0.0002, -0.0004, 0, 0, 0, 1]])
        self.assertLess(np.linalg.norm([0.0002, 0.0002, -0.0004]), 2 * 0.00028)
        return x, state, forces

    def test_rows_beyond_the_live_count_at_the_origin_equal_storage_of_exactly_the_live_size(self):
        for openings in (0, 1, 2):
            with self.subTest(openings=openings):
                x, state, forces = self.near_origin(openings)
                self.assertEqual((x.opening_capacity, x.opening_count), (16, openings))
                helper = _NeedleTrialForces(x)
                host, device = host_trial(x, state, forces), device_trial(x, helper, state, forces)
                # The tip presses on the tissue beside the origin: the kernels did work.
                self.assertGreater(float(host["diagnostics"][6, 0]), 0)
                self.assertGreater(float(device["diagnostics"][6, 0]), 0)
                self.assertEqual(int(device["fallback"][REUSE]), 0)
                # The helper computed the current position of every live opening.
                where = [weights @ VERTICES[nodes].astype(float) for nodes, weights in x._openings]
                np.testing.assert_allclose(
                    x._opening_points.numpy()[:openings], np.reshape(where, (-1, 3)), rtol=1e-6, atol=1e-12
                )
                if openings:
                    # Storage of exactly the live size: a reserve of as many rows as there are openings.
                    y, exact_state, exact_forces = self.near_origin(openings, opening_capacity=openings)
                    self.assertEqual(y._all_opening_faces.shape[0], openings)
                    self.assert_same(host, host_trial(y, exact_state, exact_forces), "host")
                    exact = device_trial(y, _NeedleTrialForces(y), exact_state, exact_forces)
                    self.assert_same(device, exact, "device")
                    np.testing.assert_array_equal(
                        x._opening_points.numpy()[:openings], y._opening_points.numpy(), "opening positions"
                    )
                else:
                    # Without an opening the two opening kernels were not launched at all before.
                    absent = device_trial(
                        x, helper, state, forces, skip=("opening_reuse_certificate", "opening_positions")
                    )
                    self.assert_same(device, absent, "device")
                # Whatever stands in the rows beyond the live count, no kernel reads it. Here: every
                # face open, at a point that is the origin for the helper and the tip for the host path.
                faces, nodes, weights = (
                    x._all_opening_faces.numpy(),
                    x._opening_nodes.numpy(),
                    x._opening_bary_d.numpy(),
                )
                points = x._opening_points.numpy()
                faces[openings:], nodes[openings:], weights[openings:] = 1, 0, (1.0, 0.0, 0.0)
                points[openings:] = (0.0002, 0.0002, -0.0004)
                for storage, values in (
                    (x._all_opening_faces, faces),
                    (x._opening_nodes, nodes),
                    (x._opening_bary_d, weights),
                    (x._opening_points, points),
                ):
                    storage.assign(values)
                self.assert_same(host_trial(x, state, forces), host, "host, other values beyond the live count")
                self.assert_same(
                    device_trial(x, helper, state, forces), device, "device, other values beyond the live count"
                )
                # The same rows, once they are live, do change the result: the comparison can fail.
                live = x._live.numpy()
                live[LIVE_OPENINGS] = x.opening_capacity
                x._live.assign(live)
                x._live_written = None
                with patch.object(type(x), "_sync_visited", lambda self: None):
                    read = device_trial(x, helper, state, forces)
                self.assertEqual(int(read["fallback"][REUSE]), 1)
                self.assertNotEqual(read["particle_f"].tobytes(), device["particle_f"].tobytes())

    @unittest.skipUnless(DEVICE.is_cuda, "CUDA graph replay contract")
    def test_graph_captured_before_an_opening_reads_the_opening_registered_after_it(self):
        x, state, forces = self.near_origin(0)
        helper = _NeedleTrialForces(x)
        # As written first: the kernel modules are loaded, and there is no opening.
        absent = device_trial(x, helper, state, forces)
        self.assertEqual((int(absent["fallback"][0]), int(absent["fallback"][REUSE])), (0, 0))
        with wp.ScopedCapture(device=DEVICE) as capture:
            state.body_f.zero_()
            forces.zero_()
            helper.add_forces(state, 0, np.zeros(3), forces, DT)

        def replay():
            wp.capture_launch(capture.graph)
            # The graph refitted the device geometry: the host path must not trust its cached positions.
            x._device_query_positions = None
            return {
                "particle_f": forces.numpy().copy(),
                "body_f": state.body_f.numpy().copy(),
                "diagnostics": x._diagnostics.numpy().copy(),
                "contact_flags": x._contact_flags.numpy().copy(),
                "mask": helper.mask.numpy().copy(),
                "fallback": helper.fallback.numpy().copy(),
            }

        self.assert_same(replay(), absent, "replay without an opening")
        # An opening 0.4 mm from the tried tip, inside the capture radius of 1 mm, registered after
        # the capture: one row written in place and the live count on the device.
        face = bottom_face(x)
        ids = x.surface_triangles[face]
        x._register_opening(ids, np.asarray([{0: 0.96, 1: 0.02, 2: 0.02}[int(v)] for v in ids]), face)
        self.assertFalse(helper.prepare(x))
        replayed = replay()
        # The graph that was captured without it asks for the host path because of it ...
        self.assertEqual((int(replayed["fallback"][0]), int(replayed["fallback"][REUSE])), (1, 1))
        self.assertNotEqual(replayed["particle_f"].tobytes(), absent["particle_f"].tobytes())
        # ... and gives what the same evaluation gives when it is issued as written with the opening.
        self.assert_same(replayed, device_trial(x, helper, state, forces), "replay with the opening")

    def test_a_registration_writes_one_row_in_place_without_new_storage_or_a_device_read(self):
        x, state, forces = self.near_origin(1)
        helper = _NeedleTrialForces(x)
        signature = helper.signature()
        arrays = {name: getattr(x, name) for name in STORAGE}
        pointers = {name: array.ptr for name, array in arrays.items()}
        face = bottom_face(x)
        ids = x.surface_triangles[face]
        allocator = DEVICE.get_allocator()
        rows = [np.asarray([0.2, 0.3, 0.5]), np.asarray([0.25, 0.5, 0.25]), np.asarray([0.5, 0.125, 0.375])]
        with (
            patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate,
            patch.object(wp.array, "numpy", side_effect=AssertionError("a registration read the device")),
        ):
            masks = [x._register_opening(ids, weights, face) for weights in rows]
            x._visited_tets.add(0)
            x._sync_visited()
        self.assertEqual(allocate.call_count, 0)
        for name, array in arrays.items():
            self.assertIs(getattr(x, name), array, name)
            self.assertEqual(array.ptr, pointers[name], name)
        self.assertEqual((x.opening_count, x.opening_capacity, x.opening_storage_growths), (4, 16, 0))
        # The helper saw no change of storage, and its signature has no entry for the counts.
        self.assertFalse(helper.prepare(x))
        self.assertEqual(helper.signature(), signature)
        np.testing.assert_array_equal(x._live.numpy(), [1, 0, 4])
        np.testing.assert_array_equal(x._visited.numpy(), [0])
        # Each row holds what was registered; the rows beyond are still zero.
        np.testing.assert_array_equal(x._all_opening_faces.numpy()[1:4], masks)
        np.testing.assert_array_equal(x._opening_nodes.numpy()[1:4], [ids] * 3)
        self.assertEqual(x._opening_bary_d.numpy()[1:4].tobytes(), np.asarray(rows).tobytes())
        for name in ("_all_opening_faces", "_opening_nodes", "_opening_bary_d"):
            self.assertFalse(getattr(x, name).numpy()[4:].any(), name)
        for (nodes, weights), mask, row in zip(x._openings[1:], x._opening_masks[1:], range(1, 4), strict=True):
            np.testing.assert_array_equal(nodes, x._opening_nodes.numpy()[row])
            np.testing.assert_array_equal(weights, x._opening_bary_d.numpy()[row])
            np.testing.assert_array_equal(mask, x._all_opening_faces.numpy()[row])
        # The visited count on the device is the number of ids that were uploaded. An edit of the host
        # set that no preparation has seen is not counted by another write of the history counts.
        x._visited_tets.discard(0)
        x._register_opening(ids, rows[0], face)
        np.testing.assert_array_equal(x._live.numpy(), [1, 0, 5])
        self.assertFalse(helper.prepare(x))
        np.testing.assert_array_equal(x._live.numpy(), [0, 0, 5])

    def test_reserve_of_one_opening_grows_by_counted_doublings_with_the_results_of_a_large_reserve(self):
        small, large = build((0.002, 0.002, -0.0001), opening_capacity=1), build((0.002, 0.002, -0.0001))
        helpers = [_NeedleTrialForces(small[0]), _NeedleTrialForces(large[0])]
        signatures = [helper.signature() for helper in helpers]
        kept = {name: getattr(large[0], name).ptr for name in STORAGE}
        grown_at = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for step, position in enumerate(three_punctures()):
                before = [
                    (x.opening_storage_growths, x.track_storage_growths, x.opening_count) for x, _, _ in (small, large)
                ]
                host = [accept(x, state, forces, position) for x, state, forces in (small, large)]
                self.assert_same(host[0], host[1], f"step {step}, host trial")
                changed = [helper.prepare(x) for helper, (x, _, _) in zip(helpers, (small, large), strict=True)]
                for index, (x, _, _) in enumerate((small, large)):
                    growths, track_growths, openings = before[index]
                    # Storage changed with a growth, never with an opening or a track sample alone.
                    grew = x.opening_storage_growths != growths
                    self.assertEqual(x.track_storage_growths, track_growths, f"step {step}")
                    self.assertEqual(changed[index], grew, f"step {step}")
                    if not grew:
                        self.assertEqual(helpers[index].signature(), signatures[index], f"step {step}")
                    signatures[index] = helpers[index].signature()
                    if grew:
                        self.assertEqual(x.opening_count, openings + 1)
                        grown_at.append((index, step, x.opening_count, x.opening_capacity))
                device = [
                    device_trial(x, helper, state, forces)
                    for helper, (x, state, forces) in zip(helpers, (small, large), strict=True)
                ]
                self.assert_same(device[0], device[1], f"step {step}, device trial")
                x, y = small[0], large[0]
                self.assertEqual(self.history(x), self.history(y), f"step {step}")
                count = x.opening_count
                np.testing.assert_array_equal(x._live.numpy(), y._live.numpy())
                for name in ("_all_opening_faces", "_opening_points", "_opening_nodes", "_opening_bary_d"):
                    a, b = getattr(x, name).numpy(), getattr(y, name).numpy()
                    self.assertEqual(a[:count].tobytes(), b[:count].tobytes(), f"step {step}: {name}")
                    if name != "_opening_points":
                        self.assertFalse(a[count:].any() or b[count:].any(), f"step {step}: {name}")
        x, y = small[0], large[0]
        self.assertEqual((y.puncture_count, y.entry_count, y.exit_count, y.opening_count), (3, 3, 2, 3))
        # The second and the third opening each doubled the reserve of one row; both were counted.
        self.assertEqual((x.opening_capacity, x.opening_storage_growths), (4, 2))
        self.assertEqual([entry[0] for entry in grown_at], [0, 0])
        self.assertEqual([entry[2:] for entry in grown_at], [(2, 2), (3, 4)])
        self.assertEqual(x._all_opening_faces.shape, (4, len(x.surface_triangles)))
        self.assertEqual(
            [str(warning.message).split(";")[0] for warning in caught],
            [
                "Needle opening storage grew from 1 to 2 rows at 1 openings",
                "Needle opening storage grew from 2 to 4 rows at 2 openings",
            ],
        )
        # The large reserve never changed its storage.
        self.assertEqual((y.opening_capacity, y.opening_storage_growths), (16, 0))
        self.assertEqual(kept, {name: getattr(y, name).ptr for name in STORAGE})

    @staticmethod
    def history(x):
        """What the accepted steps committed, apart from the storage that differs in its row count."""
        return (
            (x.status, x.reason, x.punctured, x.puncture_count, x.entry_count, x.exit_count, x.track_count),
            x.track_length,
            [(event["kind"], event["position"].tobytes(), event["reused"]) for event in x.events],
            (x.tip_inside, x._in_material, x._ever_inside, sorted(x._visited_tets)),
            [(ids.tobytes(), weights.tobytes()) for ids, weights in x._openings],
            [mask.tobytes() for mask in x._opening_masks],
            tuple(
                getattr(x, name).numpy().tobytes()
                for name in ("_track_nodes", "_track_bary", "_track_lengths", "_track_engaged")
            ),
        )

    def test_reserve_is_validated_and_reported(self):
        with self.assertRaisesRegex(ValueError, "opening_capacity must be at least one"):
            build((0.002, 0.002, -0.0001), opening_capacity=0)
        x, _, _ = build((0.002, 0.002, -0.0001), opening_capacity=3)
        self.assertEqual((x.opening_capacity, x.opening_count, x.opening_storage_growths), (3, 0, 0))
        self.assertEqual(x._all_opening_faces.shape, (3, len(x.surface_triangles)))
        self.assertEqual((x._opening_nodes.dtype, x._opening_bary_d.dtype), (wp.vec3i, wp.vec3d))
        self.assertEqual(x._visited.shape, (len(x.tetrahedra),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
