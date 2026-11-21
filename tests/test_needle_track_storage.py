# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The track samples live in storage that the interaction makes once and writes in place.

The storage has ``track_capacity`` rows, and so have the scratch of the trial-force helper and
the tunnel multipliers. Kernels that write history run over the live rows; the kernels of a
trial run over the reserve and read their bound from the live counts on the device. A row from
the live count on holds zeros: nodes 0 and weights 0, which is a material point at the world
origin, engagement 0, length 0 and multipliers 0. These tests lay a track through the world
origin, where such a row would matter if a kernel read or wrote it, and compare reserves of
different sizes. A reserve that is too small doubles, is counted, and gives the results of a
reserve that was large enough.
"""

import os
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_needle_insertion_example import _example
from test_needle_opening_storage import device_trial, host_trial, three_punctures

from newton_solver_suturebot import NeedleTissueInteraction, boundary_faces
from newton_solver_suturebot._src.solvers.coupled.interface import (
    insertion,
    needle_trial_forces,
    shaft_projected_kernels,
    shaft_projection,
    shaft_projection_tables,
    surface_contact_forces,
    tunnel_alm,
)
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.solvers.coupled.interface.needle_trial_forces import _NeedleTrialForces
from newton_solver_suturebot.examples.needle._needle_insertion_geometry import InsertionPlan

DEVICE = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
VERTICES = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], np.float64)
DT = 0.0005
# The needle of the fixture: 10 mm long, so the default reserve is floor(10 mm / 0.3 mm) + 8 rows.
DEFAULT_ROWS = 41
TRACK = ("_track_nodes", "_track_bary", "_track_bary_d", "_track_lengths", "_track_engaged", "_track_points")
MULTIPLIERS = ("committed_normal", "committed_tangent", "normal", "tangent", "proposed_normal", "proposed_tangent")
SCRATCH = ("points", "segments", "projected_arc", "associated")
# The first puncture of three_punctures is at (2, 2) mm. Moved by this offset its needle axis is
# the z axis and the world origin lies on it, 0.6 mm below the face that the needle enters.
THROUGH_ORIGIN = np.array([0.002, 0.002, 0.0006])


def build(position, *, offset=(0.0, 0.0, 0.0), **options):
    """One tetrahedron and a needle below its face, tip first; ``offset`` moves both against the world."""
    tets = np.array([[0, 1, 2, 3]], np.int32)
    vertices = VERTICES - np.asarray(offset)
    samples = np.zeros((11, 3))
    samples[:, 2] = np.linspace(0, -0.01, 11)
    interaction = NeedleTissueInteraction(
        vertices, tets, boundary_faces(tets)[0], (), samples, 0.0005, samples[0], device=DEVICE, **options
    )
    state = SimpleNamespace(
        particle_q=wp.array(vertices, dtype=wp.vec3, device=DEVICE),
        particle_qd=wp.zeros(4, dtype=wp.vec3, device=DEVICE),
        body_q=wp.array([[*(np.asarray(position) - offset), 0, 0, 0, 1]], dtype=wp.transform, device=DEVICE),
        body_qd=wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE),
        body_f=wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE),
    )
    return interaction, state, wp.zeros(4, dtype=wp.vec3, device=DEVICE)


def live(interaction, array):
    """The rows of a per-track array that hold track samples."""
    return array.numpy()[: interaction.track_count]


def helper_rows(interaction, helper):
    """What a device trial computed per track row and per needle sample; equal bits on every device."""
    rows = {name: live(interaction, getattr(helper, name)).copy() for name in ("projected_arc", "associated")}
    rows["mask"], rows["fallback"] = helper.mask.numpy().copy(), helper.fallback.numpy().copy()
    alm = interaction.tunnel_alm
    if alm is not None:
        rows["proposed_normal"] = live(interaction, alm.proposed_normal).copy()
        rows["proposed_tangent"] = live(interaction, alm.proposed_tangent).copy()
    return rows


def advance(interaction, helper, state, forces, position, *, offset=(0.0, 0.0, 0.0)):
    """One accepted implicit step to ``position``: device trial, host trial, resolution check, acceptance.

    Returns the storage change that the helper's preparation reported, the helper's signature after
    that preparation, and the outputs of both trials.
    """
    alm = interaction.tunnel_alm
    start = SimpleNamespace(particle_q=wp.clone(state.particle_q), body_q=wp.clone(state.body_q))
    state.body_q.assign([[*(np.asarray(position) - offset), 0, 0, 0, 1]])
    if alm is not None:
        alm.begin_step()
    changed, signature = helper.prepare(interaction), helper.signature()
    device = device_trial(interaction, helper, state, forces)
    rows = helper_rows(interaction, helper)
    host = host_trial(interaction, state, forces)
    interaction.check_step_resolution(start, state, 0)
    interaction.retain_trial_forces(start, 0)
    interaction.accept(state, 0)
    if interaction.status == "failed":
        raise AssertionError(interaction.reason)
    if alm is not None:
        alm.commit()
    return changed, signature, device, rows, host


def advance_explicit(interaction, state, forces, position, *, offset=(0.0, 0.0, 0.0)):
    """One accepted step of the explicit force exchange: forces at the start state, then the moved needle."""
    state.body_f.zero_()
    forces.zero_()
    interaction.add_forces(state, 0, np.zeros(3), forces, DT)
    result = {"particle_f": forces.numpy().copy(), "body_f": state.body_f.numpy().copy()}
    state.body_q.assign([[*(np.asarray(position) - offset), 0, 0, 0, 1]])
    interaction.accept(state, 0)
    if interaction.status == "failed":
        raise AssertionError(interaction.reason)
    return result


def history(interaction):
    """What the accepted steps committed; of the track storage and the multipliers the live rows."""
    x = interaction
    alm = x.tunnel_alm
    return (
        (x.status, x.reason, x.punctured, x.puncture_count, x.entry_count, x.exit_count, x.track_count),
        (x.track_length, x.engaged_track_length, tuple(map(int, x._track_sections)), tuple(sorted(x._track_edges))),
        [(event["kind"], event["position"].tobytes(), event["reused"]) for event in x.events],
        (x.tip_inside, x._in_material, x._ever_inside, sorted(x._visited_tets), len(x._openings)),
        tuple(live(x, getattr(x, name)).tobytes() for name in TRACK[:5]),
        () if alm is None else tuple(live(x, getattr(alm, name)).tobytes() for name in MULTIPLIERS),
        tuple(x._live.numpy()),
    )


def quadrature(interaction):
    """The quadrature lengths of the live rows: the host's float64 sum over the track edges, rounded once."""
    lengths = np.zeros(interaction.track_count)
    for a, b in interaction._track_edges:
        lengths[[a, b]] += np.linalg.norm(interaction._track_rest[a] - interaction._track_rest[b]) / 2
    return lengths.astype(np.float32)


def embedded(index):
    """Nodes and weights of a material point of the one tetrahedron; different points are millimetres apart."""
    weights = np.full(4, 0.1)
    weights[index % 4] = 0.7 - 0.05 * (index // 4)
    weights[(index + 1) % 4] += 0.05 * (index // 4)
    return np.array([0, 1, 2, 3], np.int32), weights


class TrackStorageTests(unittest.TestCase):
    def assert_same(self, actual, expected, label):
        """Bit for bit on the CPU. On CUDA sums over needle samples have no fixed order: integers stay exact."""
        self.assertEqual(actual.keys(), expected.keys(), label)
        for name, value in actual.items():
            if DEVICE.is_cuda and value.dtype.kind == "f":
                np.testing.assert_allclose(value, expected[name], rtol=2e-6, atol=1e-10, err_msg=f"{label}: {name}")
            else:
                self.assertEqual(value.tobytes(), expected[name].tobytes(), f"{label}: {name}")

    def assert_bits(self, actual, expected, label):
        self.assertEqual(actual.keys(), expected.keys(), label)
        for name, value in actual.items():
            self.assertEqual(value.tobytes(), expected[name].tobytes(), f"{label}: {name}")

    def assert_rows_beyond_are_zero(self, interaction, label, helper=None):
        """The invariant of the reserve: every row from the live count on holds zeros."""
        count = interaction.track_count
        arrays = {name: getattr(interaction, name) for name in TRACK[:5]}
        if interaction.tunnel_alm is not None:
            arrays.update({name: getattr(interaction.tunnel_alm, name) for name in MULTIPLIERS})
        if helper is not None:
            arrays.update({name: getattr(helper, name) for name in ("segments", "projected_arc", "associated")})
        for name, array in arrays.items():
            self.assertEqual(array.shape[0], interaction.track_capacity, f"{label}: {name}")
            self.assertFalse(array.numpy()[count:].any(), f"{label}: {name} beyond {count} rows")

    def run_path(self, path, *, offset=(0.0, 0.0, 0.0), **options):
        """Accepted implicit steps along ``path``; the invariant of the reserve is checked after every step."""
        x, state, forces = build(path[0], offset=offset, **options)
        helper = _NeedleTrialForces(x)
        steps = []
        for step, position in enumerate(path):
            grown = x.track_storage_growths
            changed, signature, device, rows, host = advance(x, helper, state, forces, position, offset=offset)
            steps.append(
                {
                    "grown": grown,
                    "changed": changed,
                    "signature": signature,
                    "device": device,
                    "rows": rows,
                    "host": host,
                    "history": history(x),
                    "tip": x._previous_tip.copy(),
                    "engaged": int(live(x, x._track_engaged).sum()),
                    "multiplier": 0.0
                    if x.tunnel_alm is None or not x.track_count
                    else float(np.abs(live(x, x.tunnel_alm.committed_normal)).max()),
                }
            )
            # After a growth in this step's acceptance the helper's scratch follows at its next preparation.
            current = helper.projected_arc.shape[0] == x.track_capacity
            self.assert_rows_beyond_are_zero(x, f"step {step}", helper if current else None)
        return x, helper, steps

    def assert_equal_runs(self, steps, reference, label):
        self.assertEqual(len(steps), len(reference), label)
        for step, (actual, expected) in enumerate(zip(steps, reference, strict=True)):
            self.assert_same(actual["device"], expected["device"], f"{label}, step {step}: device trial")
            self.assert_bits(actual["rows"], expected["rows"], f"{label}, step {step}: rows of the device trial")
            self.assert_same(actual["host"], expected["host"], f"{label}, step {step}: host trial")
            self.assertEqual(actual["history"], expected["history"], f"{label}, step {step}")

    def test_reserve_is_derived_from_the_needle_validated_and_reported(self):
        x, _, _ = build((0.002, 0.002, -0.0001), tunnel_model="alm")
        # floor(needle length / track spacing) samples of one passage, and eight more.
        self.assertEqual(int(np.floor(x._sample_arc[-1] / x.track_spacing)) + 8, DEFAULT_ROWS)
        self.assertEqual((x.track_capacity, x.track_count, x.track_storage_growths), (DEFAULT_ROWS, 0, 0))
        types = (wp.vec4i, wp.vec4, wp.vec4d, wp.float32, wp.int32, wp.vec3)
        for name, dtype in zip(TRACK, types, strict=True):
            array = getattr(x, name)
            self.assertEqual((array.shape, array.dtype), ((DEFAULT_ROWS,), dtype), name)
            self.assertFalse(array.numpy().any(), name)
        for name in MULTIPLIERS:
            array = getattr(x.tunnel_alm, name)
            self.assertEqual(array.shape, (DEFAULT_ROWS,), name)
            self.assertFalse(array.numpy().any(), name)
        helper = _NeedleTrialForces(x)
        self.assertIs(helper.nodes, x._track_nodes)
        self.assertIs(helper.bary, x._track_bary_d)
        shapes = ((DEFAULT_ROWS,), (DEFAULT_ROWS, len(x.samples) - 1), (DEFAULT_ROWS,), (DEFAULT_ROWS,))
        for name, shape in zip(SCRATCH, shapes, strict=True):
            self.assertEqual(getattr(helper, name).shape, shape, name)
        self.assertEqual(helper.signature()[0], (DEFAULT_ROWS, x.opening_capacity))
        np.testing.assert_array_equal(x._live.numpy(), 0)
        # The storage that the helper's launches address is part of its signature: the live counts, the
        # track rows in both precisions and the scratch per track row.
        signature = helper.signature()
        for owner, name in ((x, "_live"), (x, "_track_nodes"), (x, "_track_bary_d"), (helper, "points")):
            kept = getattr(owner, name)
            setattr(owner, name, wp.clone(kept))
            self.assertNotEqual(helper.signature(), signature, name)
            setattr(owner, name, kept)
            self.assertEqual(helper.signature(), signature, name)
        # The keyword is the reserve; maximum_track_points stays the limit of the track.
        y, _, _ = build((0.002, 0.002, -0.0001), track_capacity=5)
        self.assertEqual((y.track_capacity, y._track_nodes.shape, y._track_bary_d.shape), (5, (5,), (5,)))
        z, _, _ = build((0.002, 0.002, -0.0001), track_capacity=500, maximum_track_points=64)
        self.assertEqual((z.track_capacity, z._track_engaged.shape), (64, (64,)))
        short, _, _ = build((0.002, 0.002, -0.0001), maximum_track_points=16)
        self.assertEqual(short.track_capacity, 16)
        with self.assertRaisesRegex(ValueError, "track_capacity must be at least one"):
            build((0.002, 0.002, -0.0001), track_capacity=0)

    def test_an_appended_sample_writes_its_row_in_place_without_new_storage_or_a_device_read(self):
        for model in ("penalty", "alm"):
            with self.subTest(model=model):
                x, _, _ = build((0.002, 0.002, -0.0001), tunnel_model=model)
                helper = _NeedleTrialForces(x)
                signature = helper.signature()
                owners = {name: getattr(x, name) for name in (*TRACK, "_live")}
                owners.update({f"helper.{name}": getattr(helper, name) for name in (*SCRATCH, "nodes", "bary")})
                if x.tunnel_alm is not None:
                    owners.update({f"tunnel.{name}": getattr(x.tunnel_alm, name) for name in MULTIPLIERS})
                pointers = {name: array.ptr for name, array in owners.items()}
                allocator = DEVICE.get_allocator()
                rows = [embedded(index) for index in range(3)]
                with (
                    patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate,
                    patch.object(wp.array, "numpy", side_effect=AssertionError("an append read the device")),
                ):
                    for nodes, weights in rows:
                        x._record_track(nodes, weights)
                self.assertEqual(allocate.call_count, 0)
                self.assertEqual((x.track_count, x.track_capacity, x.track_storage_growths), (3, DEFAULT_ROWS, 0))
                # The helper saw no change of storage, and its signature has no entry for the count.
                self.assertFalse(helper.prepare(x))
                self.assertEqual(helper.signature(), signature)
                current = {name: getattr(x, name) for name in (*TRACK, "_live")}
                current.update({f"helper.{name}": getattr(helper, name) for name in (*SCRATCH, "nodes", "bary")})
                if x.tunnel_alm is not None:
                    current.update({f"tunnel.{name}": getattr(x.tunnel_alm, name) for name in MULTIPLIERS})
                for name, array in owners.items():
                    self.assertIs(current[name], array, name)
                    self.assertEqual(array.ptr, pointers[name], name)
                np.testing.assert_array_equal(x._live.numpy(), [0, 3, 0])
                # Each row holds what was recorded: the weights rounded to single precision for the
                # force kernels and unrounded for the helper, the engagement of a new sample, and
                # the quadrature lengths of the host's float64 sum, rounded once.
                weights = np.asarray([row[1] for row in rows])
                np.testing.assert_array_equal(live(x, x._track_nodes), [row[0] for row in rows])
                self.assertEqual(live(x, x._track_bary).tobytes(), weights.astype(np.float32).tobytes())
                self.assertEqual(live(x, x._track_bary_d).tobytes(), weights.tobytes())
                np.testing.assert_array_equal(live(x, x._track_engaged), 1)
                lengths = quadrature(x)
                self.assertEqual(sorted(x._track_edges), [(0, 1), (1, 2)])
                self.assertEqual(live(x, x._track_lengths).tobytes(), lengths.tobytes())
                self.assertGreater(float(lengths.min()), 0)
                self.assert_rows_beyond_are_zero(x, "after three samples", helper)
                # An older row keeps what the device holds: a released sample is not engaged again
                # by the next append, and the lengths of all live rows follow the new edge.
                released = np.array([0, 1, 0], np.int32)
                wp.copy(x._track_engaged, wp.array(released, dtype=int, device="cpu"), count=3)
                x._engaged_length = (x._track_engaged, x._track_lengths, 123.0)
                x._record_track(*embedded(3))
                np.testing.assert_array_equal(live(x, x._track_engaged), [0, 1, 0, 1])
                self.assertEqual(sorted(x._track_edges), [(0, 1), (1, 2), (2, 3)])
                self.assertEqual(live(x, x._track_lengths).tobytes(), quadrature(x).tobytes())
                self.assertGreater(float(live(x, x._track_lengths)[2]), float(lengths[2]))
                # The append wrote engagement and lengths: a length read before it is not served.
                expected = float(np.sum(live(x, x._track_lengths)[live(x, x._track_engaged) != 0]))
                self.assertEqual(x.engaged_track_length, expected)
                self.assertNotEqual(expected, 123.0)

    def test_a_small_reserve_grows_by_counted_doublings_with_the_results_of_a_large_reserve(self):
        path = three_punctures()
        for model in ("penalty", "alm"):
            with self.subTest(model=model):
                scene, _, reference = self.run_path(path, tunnel_model=model)
                kept = {name: getattr(scene, name).ptr for name in TRACK}
                count = scene.track_count
                self.assertEqual((scene.puncture_count, scene.entry_count, scene.exit_count), (3, 3, 2))
                self.assertGreater(count, 8)
                self.assertEqual((scene.track_capacity, scene.track_storage_growths), (DEFAULT_ROWS, 0))
                self.assertLess(count, DEFAULT_ROWS)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    small, _, steps = self.run_path(path, tunnel_model=model, track_capacity=2)
                self.assert_equal_runs(steps, reference, "reserve of 2 rows")
                # The small reserve doubled until it held the track; every doubling was counted and warned about.
                doublings = int(np.ceil(np.log2(count / 2)))
                self.assertGreater(doublings, 1)
                self.assertEqual((small.track_capacity, small.track_storage_growths), (2 * 2**doublings, doublings))
                self.assertEqual(
                    [str(warning.message).split(";")[0] for warning in caught],
                    [
                        f"Needle track storage grew from {2 * 2**k} to {4 * 2**k} rows at {2 * 2**k} samples"
                        for k in range(doublings)
                    ],
                )
                # The helper's storage changed in the step after a growth, never with a sample alone, and
                # its signature changed with it and at no other step.
                seen, signature = 0, steps[0]["signature"]
                for step, record in enumerate(steps):
                    self.assertEqual(record["changed"], record["grown"] != seen, f"step {step}")
                    self.assertEqual(record["signature"] != signature, record["changed"], f"step {step}")
                    seen, signature = record["grown"], record["signature"]
                self.assertEqual(sum(record["changed"] for record in steps), doublings)
                large, _, steps = self.run_path(path, tunnel_model=model, track_capacity=512)
                self.assert_equal_runs(steps, reference, "reserve of 512 rows")
                # The reserves that were large enough never changed their storage or their helper's signature.
                self.assertEqual((large.track_capacity, large.track_storage_growths), (512, 0))
                for records in (reference, steps):
                    self.assertFalse(any(record["changed"] for record in records))
                    self.assertEqual({record["signature"] for record in records}, {records[0]["signature"]})
                self.assertEqual(kept, {name: getattr(scene, name).ptr for name in TRACK})

    def test_committed_tunnel_multipliers_survive_a_growth_of_the_reserve(self):
        # Enter, press the shaft sideways against the tunnel past its clearance, then go on while pressed:
        # the samples that follow double a reserve of two rows while the first rows hold reactions.
        path = [(0.002, 0.002, z) for z in (-0.0001, 0.0, 0.00015, 0.0003, 0.00045)]
        path += [(0.002 + shift, 0.002, 0.00045) for shift in (2e-5, 4e-5, 6e-5, 8e-5)]
        path += [(0.00208, 0.002, z) for z in np.arange(4, 15) * 0.00015]
        scene, _, reference = self.run_path(path, tunnel_model="alm")
        self.assertEqual((scene.track_capacity, scene.track_storage_growths), (DEFAULT_ROWS, 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small, _, steps = self.run_path(path, tunnel_model="alm", track_capacity=2)
        self.assert_equal_runs(steps, reference, "reserve of 2 rows")
        self.assertGreater(small.track_storage_growths, 1)
        self.assertEqual(small.track_count, scene.track_count)
        # Every growth copied rows that held a reaction [N/m], and the reactions are those of the large reserve.
        grown = [step for step in range(1, len(steps)) if steps[step]["grown"] != steps[step - 1]["grown"]]
        grown += [len(steps)] if small.track_storage_growths != steps[-1]["grown"] else []
        self.assertEqual(len(grown), small.track_storage_growths)
        for step in grown:
            # The growth happened in the acceptance of the step before; its committed rows were pressed.
            self.assertGreater(steps[step - 2]["multiplier"], 0.05, f"growth before step {step}")
        self.assertGreater(steps[-1]["multiplier"], 0.05)
        self.assertEqual(steps[-1]["multiplier"], reference[-1]["multiplier"])

    def test_track_through_the_world_origin_leaves_the_rows_beyond_the_live_count_alone(self):
        offset = THROUGH_ORIGIN
        # Entry on the axis through the origin, 1.2 mm deep, and back out: the shaft lies on the origin.
        path = three_punctures()[:26]
        for model in ("penalty", "alm"):
            with self.subTest(model=model):
                x, _, reference = self.run_path(path, offset=offset, tunnel_model=model)
                count = x.track_count
                self.assertGreater(count, 3)
                self.assertGreater(x.track_capacity, count)
                self.assertEqual((x.track_storage_growths, x.entry_count, x.exit_count), (0, 1, 1))
                # At the deepest step the tip is 0.6 mm past the origin: the origin lies on the engaged shaft,
                # inside the tissue and within the capture radius of the needle samples around it.
                deepest = reference[9]
                self.assertLess(float(np.linalg.norm(deepest["tip"][:2])), 1e-9)
                self.assertGreater(float(deepest["tip"][2]), 0.00055)
                self.assertGreater(deepest["engaged"], 2)
                self.assertEqual(reference[-1]["engaged"], 0)
                # Storage of exactly the final size: no row beyond the live count is left at the end.
                y, _, steps = self.run_path(path, offset=offset, tunnel_model=model, track_capacity=count)
                self.assertEqual((y.track_count, y.track_capacity, y.track_storage_growths), (count, count, 0))
                self.assert_equal_runs(steps, reference, "storage of exactly the final size")

    def test_values_in_rows_beyond_the_live_count_are_not_read_until_the_count_reaches_them(self):
        offset = THROUGH_ORIGIN
        path = three_punctures()[:11]
        x, state, forces = build(path[0], offset=offset)
        helper = _NeedleTrialForces(x)
        for position in path:
            advance(x, helper, state, forces, position, offset=offset)
        count = x.track_count
        self.assertGreater(count, 2)
        self.assertGreater(x.track_capacity, count)
        self.assertEqual((x.status, x.tunnel_alm), ("inserted", None))
        state.body_q.assign([[3e-5, 0, float(x._previous_tip[2]), 0, 0, 0, 1]])
        host, device = host_trial(x, state, forces), device_trial(x, helper, state, forces)
        rows = helper_rows(x, helper)
        self.assertGreater(float(np.abs(device["particle_f"]).max()), 0)
        # Every row beyond the live count as a sample 0.2 mm beside the shaft, engaged over a metre of track.
        tip = np.asarray(x._previous_tip) + (0, 2e-4, 0)
        weights = np.linalg.solve(np.vstack((x.vertices.T, np.ones(4))), np.r_[tip, 1.0])
        self.assertGreater(float(weights.min()), 0)
        poison = {
            "_track_nodes": [0, 1, 2, 3],
            "_track_bary": weights.astype(np.float32),
            "_track_bary_d": weights,
            "_track_lengths": 1.0,
            "_track_engaged": 1,
            "_track_points": tip.astype(np.float32),
        }
        for name, value in poison.items():
            values = getattr(x, name).numpy()
            values[count:] = value
            getattr(x, name).assign(values)
        for name, value in (("points", tip), ("projected_arc", 0.0), ("associated", 1)):
            values = getattr(helper, name).numpy()
            values[count:] = value
            getattr(helper, name).assign(values)
        self.assert_same(host_trial(x, state, forces), host, "host, other values beyond the live count")
        self.assert_same(device_trial(x, helper, state, forces), device, "device, other values beyond the live count")
        self.assert_bits(helper_rows(x, helper), rows, "rows, other values beyond the live count")
        # The same rows, once they are live, do change the result: the comparison can fail.
        counts = x._live.numpy()
        counts[LIVE_TRACK] = x.track_capacity
        x._live.assign(counts)
        x._live_written = None
        with patch.object(type(x), "_sync_visited", lambda self: None):
            read = device_trial(x, helper, state, forces)
        self.assertNotEqual(read["particle_f"].tobytes(), device["particle_f"].tobytes())
        self.assertGreater(float(np.abs(read["particle_f"]).max()), 10 * float(np.abs(device["particle_f"]).max()))

    def test_explicit_force_exchange_writes_engagement_over_the_live_rows_only(self):
        offset = THROUGH_ORIGIN
        path = three_punctures()[:26]

        def run(**options):
            x, state, forces = build(path[0], offset=offset, **options)
            steps = []
            for step, position in enumerate(path):
                result = advance_explicit(x, state, forces, position, offset=offset)
                steps.append((result, history(x), int(live(x, x._track_engaged).sum())))
                self.assert_rows_beyond_are_zero(x, f"step {step}")
            return x, steps

        x, reference = run()
        count = x.track_count
        self.assertGreater(count, 3)
        self.assertGreater(x.track_capacity, count)
        self.assertEqual((x.track_storage_growths, x.entry_count, x.exit_count), (0, 1, 1))
        # Samples were engaged on the way in and released on the way out, each on a live row: the kernels
        # that write the engagement ran over the live rows, although the origin lies on the shaft.
        engaged = [record[2] for record in reference]
        self.assertGreater(max(engaged), 2)
        self.assertEqual(engaged[-1], 0)
        for label, options in (
            ("storage of exactly the final size", {"track_capacity": count}),
            ("2 rows", {"track_capacity": 2}),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, steps = run(**options)
            self.assertEqual(y.track_count, count, label)
            for step, (actual, expected) in enumerate(zip(steps, reference, strict=True)):
                self.assert_same(actual[0], expected[0], f"{label}, step {step}")
                self.assertEqual(actual[1], expected[1], f"{label}, step {step}")

    def test_launches_of_a_device_trial_do_not_follow_the_track_count(self):
        # The kernels over track rows and their launch sizes are those of the reserve for any number of
        # samples: a graph captured at one track count is the graph of every other.
        over_rows = {
            "track_segments",
            "track_geometry",
            "_track_positions",
            "shaft_mask",
            "_shaft_forces",
            "shaft_forces_projected",
            "_tunnel_forces",
            "tunnel_forces_projected",
            "prepare_queries",
            "cooperative_projection",
        }
        launch = wp.launch
        for model in ("penalty", "alm"):
            with self.subTest(model=model):
                x, state, forces = build(three_punctures()[0], tunnel_model=model)
                helper = _NeedleTrialForces(x)

                def trial(x=x, helper=helper, state=state, forces=forces):
                    issued = []

                    def record(kernel, dim, *arguments, **options):
                        if kernel.key in over_rows:
                            issued.append((kernel.key, tuple(np.atleast_1d(dim)), options.get("block_dim")))
                        return launch(kernel, dim, *arguments, **options)

                    if x.tunnel_alm is not None:
                        x.tunnel_alm.begin_step()
                    with patch.object(wp, "launch", side_effect=record):
                        device_trial(x, helper, state, forces)
                    if x.tunnel_alm is not None:
                        x.tunnel_alm.discard()
                    return issued

                sequences = {0: trial()}
                for position in three_punctures()[:10]:
                    advance(x, helper, state, forces, position)
                    sequences.setdefault(x.track_count, trial())
                counts = sorted(sequences)
                self.assertGreater(len(counts), 3)
                self.assertEqual(x.track_storage_growths, 0)
                rows, law = x.track_capacity, "_shaft_forces" if model == "penalty" else "_tunnel_forces"
                expected = [
                    ("_track_positions", (rows,), None),
                    ("track_segments", (rows, len(x.samples) - 1), None),
                    ("track_geometry", (rows,), None),
                    ("shaft_mask", (len(x.samples),), 1),
                    (law, (rows,), None),
                ]
                for count in counts[1:]:
                    self.assertEqual(sequences[count], expected, f"{count} samples")
                # An empty track: the same launches. The tunnel kernel alone is left out, outside a capture.
                self.assertEqual(sequences[0], expected if model == "penalty" else expected[:-1])

    def test_tunnel_copies_the_live_rows_and_nothing_at_an_empty_track(self):
        x, state, forces = build((0.002, 0.002, -0.0001), tunnel_model="alm")
        alm = x.tunnel_alm
        pointers = {name: getattr(alm, name).ptr for name in MULTIPLIERS}
        copy = wp.copy

        def copies(call):
            with patch.object(wp, "copy", side_effect=copy) as issued:
                call()
            return [arguments.kwargs.get("count") for arguments in issued.call_args_list]

        # An empty track: no copy is issued (a count of zero would mean the whole array to wp.copy),
        # and outside a capture no kernel either.
        with patch.object(wp, "launch", side_effect=AssertionError("a launch at an empty track")):
            for call in (alm.begin_step, alm.update_duals, alm.discard, alm.begin_step, alm.commit):
                self.assertEqual(copies(call), [], call.__name__)
            alm.begin_step()
            alm.add_forces(state, 0, np.zeros(3), forces, DT)
            alm.discard()
        helper = _NeedleTrialForces(x)
        for position in three_punctures()[:8]:
            advance(x, helper, state, forces, position)
        count = x.track_count
        self.assertGreater(count, 2)
        # With a track every copy moves exactly the live rows of the step.
        self.assertEqual(copies(alm.begin_step), [count] * 4)
        self.assertEqual(copies(alm.update_duals), [count] * 2)
        self.assertEqual(copies(alm.discard), [count] * 4)
        # The count of a step is the host's count at its begin: a sample appended inside it is refused.
        alm.begin_step()
        x._record_track(*embedded(1))
        self.assertEqual(x.track_count, count + 1)
        with self.assertRaisesRegex(RuntimeError, "Material tunnel history changed during an ALM trial"):
            alm.add_forces(state, 0, np.zeros(3), forces, DT)
        # A commit moves the rows of the step it closes, and the new row starts from zero multipliers.
        alm.proposed_normal.fill_(wp.vec3(1.0, 2.0, 3.0))
        alm.proposed_tangent.fill_(4.0)
        x._track_engaged.fill_(1)
        x._track_lengths.fill_(1.0)
        self.assertEqual(copies(alm.commit), [count] * 2 + [count + 1] * 4)
        for name, value in (("committed_normal", (1.0, 2.0, 3.0)), ("normal", (1.0, 2.0, 3.0))):
            values = getattr(alm, name).numpy()
            np.testing.assert_array_equal(values[:count], np.broadcast_to(value, (count, 3)), name)
            self.assertFalse(values[count:].any(), name)
        for name in ("committed_tangent", "tangent"):
            values = getattr(alm, name).numpy()
            np.testing.assert_array_equal(values[:count], 4.0, name)
            self.assertFalse(values[count:].any(), name)
        np.testing.assert_array_equal(
            alm.proposed_normal.numpy()[: count + 1], alm.committed_normal.numpy()[: count + 1]
        )
        self.assertEqual(pointers, {name: getattr(alm, name).ptr for name in MULTIPLIERS})
        self.assertEqual(alm.capacity, x.track_capacity)

    def test_maximum_track_points_stays_the_only_hard_limit(self):
        x, _, _ = build((0.002, 0.002, -0.0001), track_capacity=2, maximum_track_points=5, tunnel_model="alm")
        helper = _NeedleTrialForces(x)
        alm = x.tunnel_alm
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for index in range(5):
                x._record_track(*embedded(index))
                self.assertNotEqual(x.status, "failed")
        # 2, 4 and then the limit itself: the last growth is a growth to maximum_track_points.
        self.assertEqual((x.track_count, x.track_capacity, x.track_storage_growths), (5, 5, 2))
        self.assertEqual(len(caught), 2)
        self.assertTrue(helper.prepare(x))
        self.assertFalse(helper.prepare(x))
        self.assertEqual(helper.segments.shape, (5, len(x.samples) - 1))
        alm.begin_step()
        self.assertEqual((alm.capacity, alm.committed_normal.shape), (5, (5,)))
        alm.discard()
        np.testing.assert_array_equal(live(x, x._track_nodes), [[0, 1, 2, 3]] * 5)
        self.assertEqual(
            live(x, x._track_bary_d).tobytes(), np.asarray([embedded(index)[1] for index in range(5)]).tobytes()
        )
        np.testing.assert_array_equal(live(x, x._track_engaged), 1)
        # One sample more than the limit fails the insertion, as before, and nothing grows.
        x._record_track(*embedded(5))
        self.assertEqual((x.status, x.reason), ("failed", "Insertion track capacity exceeded"))
        self.assertEqual((x.track_count, x.track_capacity, x.track_storage_growths), (5, 5, 2))

    @unittest.skipUnless(DEVICE.is_cuda, "Kernel modules are loaded per block size on CUDA")
    def test_helper_constructor_loads_every_module_build_that_its_launches_use(self):
        from newton_solver_suturebot._src.geometry import mesh_refit, needle_queries
        from newton_solver_suturebot._src.solvers.coupled.acceleration.surface_contact_tangent import (
            SurfaceTangentAnchors,
        )

        module_type = type(insertion._surface_forces.module)
        real = module_type.load
        asked = []

        def load(module, device, block_dim=None, *arguments, **options):
            executable = real(module, device, block_dim, *arguments, **options)
            # A build belongs to one block size; without one the module's own default is loaded.
            asked.append((module.name, module.options["block_dim"] if block_dim is None else block_dim))
            return executable

        for model in ("penalty", "alm"):
            with self.subTest(model=model):
                x, state, forces = build(three_punctures()[0], tunnel_model=model)
                plain = _NeedleTrialForces(x)
                for position in three_punctures()[:8]:
                    advance(x, plain, state, forces, position)
                self.assertGreater(x.track_count, 2)
                del asked[:]
                with patch.object(module_type, "load", autospec=True, side_effect=load):
                    helper = _NeedleTrialForces(x)
                    loaded = set(asked)
                    del asked[:]
                    for projected in (False, True):
                        x.cooperative_shaft_projection = projected
                        for recorded in (False, True):
                            x._surface_tangent_anchors = SurfaceTangentAnchors(64, device=DEVICE) if recorded else None
                            x._surface_tangent_moment_length = 0.01
                            if x.tunnel_alm is not None:
                                x.tunnel_alm.discard()
                                x.tunnel_alm.begin_step()
                            # The helper's kernels, then the host path with its own block size.
                            device_trial(x, helper, state, forces)
                            host_trial(x, state, forces)
                    x._surface_tangent_anchors = None
                    x.cooperative_shaft_projection = False
                    if x.tunnel_alm is not None:
                        x.tunnel_alm.discard()
                used = set(asked)
                # Every build that a launch asked for was loaded by the constructor.
                self.assertEqual(sorted(used - loaded), [])
                names = {module.__name__ for module in (insertion, needle_trial_forces, surface_contact_forces)}
                names |= {module.__name__ for module in (mesh_refit, needle_queries, shaft_projection_tables)}
                names |= {module.__name__ for module in (shaft_projection, shaft_projected_kernels)}
                if model == "alm":
                    names.add(tunnel_alm.__name__)
                self.assertEqual(sorted(names - {name for name, _ in used}), [])
                self.assertIn((insertion.__name__, 1), used)
                self.assertIn((insertion.__name__, 256), used)

    def test_insertion_reserves_sixteen_openings_for_every_planned_section(self):
        # One passage: one section in intact tissue, one in each lip across a wound.
        for sections in (1, 2):
            with self.subTest(sections=sections), _example() as example:
                example.select_regions([1])
                example.select_point_xy("entry", -0.004, 0)
                example.select_point_xy("exit", 0.004, 0)
                example.preview()
                self.assertEqual(example.plan.expected_passages, 1)
                with patch.object(InsertionPlan, "expected_passages", new_callable=PropertyMock, return_value=sections):
                    self.assertEqual(example._planned_passages(), (1, sections))
                    reserve = example._history_reserve()
                    example.start()
                length = float(np.linalg.norm(np.diff(example.shaft_samples, axis=0), axis=1).sum())
                per_passage = int(np.floor(length / example.args.track_spacing))
                self.assertEqual(
                    reserve, {"opening_capacity": 16 * sections, "track_capacity": per_passage + 8 * sections}
                )
                x = example.interaction
                self.assertEqual((x.opening_capacity, x.track_capacity), (16 * sections, per_passage + 8 * sections))
                self.assertEqual(x._all_opening_faces.shape, (16 * sections, len(x.surface_triangles)))
                report = example.report()
                self.assertEqual(
                    (report["openings"], report["opening_capacity"], report["opening_storage_growths"]),
                    (0, 16 * sections, 0),
                )

    def test_suture_row_reserves_a_needle_length_of_samples_for_every_stitch(self):
        from newton_solver_suturebot.examples.needle import example_fem_needle_suture as suture

        args = suture.Example.create_parser().parse_args(["--stitches", "4"])
        with wp.ScopedDevice(DEVICE):
            viewer = ViewerNull()
            try:
                example = suture.Example(viewer, args)
                self.assertEqual(len(example.stitches), 4)
                passages, sections = example._planned_passages()
                self.assertEqual(passages, 4)
                length = float(np.linalg.norm(np.diff(example.shaft_samples, axis=0), axis=1).sum())
                per_passage = int(np.floor(length / args.track_spacing))
                reserve = example._history_reserve()
                self.assertEqual(reserve["track_capacity"], 4 * per_passage + 8 * sections * 4)
                self.assertEqual(reserve["opening_capacity"], 2 * 8 * sections * 4)
                example.start()
                x = example.interaction
                # More than one passage would fit into the reserve of a single insertion.
                self.assertGreater(x.track_capacity, per_passage + 8 * sections)
                self.assertEqual(x.track_capacity, reserve["track_capacity"])
                self.assertLessEqual(x.track_capacity, x.maximum_track_points)
                self.assertEqual(x._track_nodes.shape, (reserve["track_capacity"],))
                self.assertEqual(x.opening_capacity, reserve["opening_capacity"])
                if x.tunnel_alm is not None:
                    self.assertEqual(x.tunnel_alm.committed_normal.shape, (reserve["track_capacity"],))
                report = example.report()
                self.assertEqual(
                    (report["track_samples"], report["track_capacity"], report["track_storage_growths"]),
                    (0, reserve["track_capacity"], 0),
                )
                self.assertEqual(
                    (report["openings"], report["opening_capacity"], report["opening_storage_growths"]),
                    (0, reserve["opening_capacity"], 0),
                )
                self.assertEqual(report["cooperative_shaft_rebinds"], 0)
            finally:
                viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
