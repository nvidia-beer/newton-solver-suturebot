# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The anchors of the shaft correction have the rows of the track reserve.

Tangents, levers and shears are made once, with one row for every row of the interaction's track
storage. The kernels over them are launched over that reserve and read the number of track samples
from the interaction's live counts on the device: a thread beyond it returns before it reads or
writes anything. A graph that holds these launches therefore does not follow the track count, and
the count is no part of a graph key any more; the storage is.

Checked here:

- each of the seven kernels at the reserve, with rows beyond the samples that would poison every
  result, against its text without the bound launched over the samples: the same bits;
- a whole correction, and a coupled step of the native host path, in storage of exactly the track
  samples and in the padded reserve: the same bits on the CPU;
- a reserve that was too small: the anchors follow the interaction's growth once, counted, and
  give the bits of a reserve that was large enough;
- the host stages the time step and the native damping where they changed and makes nothing; an
  invalid damping and, for a track, an active tape are refused before anything is written, and the
  articulated trials keep their two refusals;
- the storage key holds the storage of everything a captured correction addresses on the object
  and not the track count; replaced storage captures the trial graph and the solve graph again,
  a new track sample captures nothing.
"""

import inspect
import linecache
import os
import re
import unittest
import warnings
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import test_needle_coupling_graph as coupled
import warp as wp
from test_combined_contact_preconditioner import native_embedded
from test_cooperative_shaft_runtime import original_packet
from test_needle_insertion_example import _example
from test_needle_shaft_preconditioner import _apply, append_engaged_sample, setup

from newton_solver_suturebot._src.solvers.coupled.acceleration import shaft_preconditioner
from newton_solver_suturebot._src.solvers.coupled.acceleration.shaft_preconditioner import _ShaftPreconditioner
from newton_solver_suturebot._src.solvers.coupled.interface import shaft_projected_kernels
from newton_solver_suturebot._src.solvers.coupled.interface.insertion import LIVE_TRACK
from newton_solver_suturebot._src.solvers.coupled.interface.tunnel_alm import _TunnelALM

DEVICE = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
H = 1 / 240

# ---------------------------------------------------------------------------------------------
# The kernels, without their bound, as they were while they were launched over the track samples.

_BOUND = ((r"    live: wp\.array\[int\],\n", ""), (r"    if i >= live\[LIVE_TRACK\]:\n        return\n", ""))
_MODULES = {
    "prepare_anchors": shaft_preconditioner,
    "prepare_alm_anchors": shaft_preconditioner,
    "contact_action": shaft_preconditioner,
    "contact_diagonal": shaft_preconditioner,
    "balanced_correction": shaft_preconditioner,
    "prepare_anchors_projected": shaft_projected_kernels,
    "prepare_alm_anchors_projected": shaft_projected_kernels,
}
_unbounded = {}


def unbounded(name):
    """Compile the text of a kernel without its bound next to the globals of its module."""
    if name not in _unbounded:
        module = _MODULES[name]
        text = inspect.getsource(getattr(module, name).func)
        for pattern, replacement in _BOUND:
            text, found = re.subn(pattern, replacement, text)
            if found != 1:
                raise AssertionError(f"{name}: {pattern!r} matches {found} places, expected 1")
        if "live" in text:
            raise AssertionError(f"{name}: the text without the bound still reads the live counts")
        namespace = dict(vars(module), __name__=f"{__name__}.{module.__name__.rsplit('.', 1)[-1]}")
        filename = f"<{namespace['__name__']}.{name}>"
        # Warp reads a kernel's source through inspect, which serves this text from the line cache.
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        exec(compile(text, filename, "exec"), namespace)
        _unbounded[name] = namespace[name]
    return _unbounded[name]


def array(values, dtype):
    return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device=DEVICE)


class Track:
    """``count`` track samples beside a straight shaft, in storage of ``capacity`` rows.

    The rows beyond the samples hold what no thread may use. Each names particle 0 with weight
    one, a quadrature length of a metre and an engagement, so nothing but the bound keeps a
    kernel away from it; particle 0 has no finite position, velocity or trial velocity, and the
    multipliers, tangents, levers, shears and projections of these rows are NaN. No track
    sample uses particle 0.
    """

    PARTICLES, PENALTY, CLEARANCE, RADIUS = 12, 60000.0, 2.0e-5, 0.0005

    def __init__(self, seed, count, capacity):
        rng = np.random.default_rng(seed)
        self.count, self.capacity = count, capacity
        line = np.stack([np.linspace(-0.01, 0.01, 21), np.zeros(21), np.zeros(21)], axis=1)
        self.samples = array(line, wp.vec3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.2, 1.2)
        quat = np.r_[np.sin(angle / 2) * axis, np.cos(angle / 2)]
        position = rng.normal(0, 0.01, 3)
        rotation = np.array(wp.quat_to_matrix(wp.quat(*quat))).reshape(3, 3)
        # Tissue points within a few clearances of the shaft, on both sides of the tunnel's threshold.
        along = rng.uniform(-0.006, 0.006, self.PARTICLES)
        offsets = rng.normal(0, 1.5e-4, (self.PARTICLES, 3))
        offsets[:, 0] = 0
        q = position + (np.stack([along, np.zeros_like(along), np.zeros_like(along)], axis=1) + offsets) @ rotation.T
        qd = rng.normal(0, 2.0e-3, (self.PARTICLES, 3))
        q[0], qd[0] = np.nan, np.inf
        self.q, self.qd = array(q, wp.vec3), array(qd, wp.vec3)
        self.poses = array(np.r_[position, quat][None], wp.transform)
        velocity = np.r_[0.004 * rotation[:, 0] + rng.normal(0, 2.0e-4, 3), rng.normal(0, 0.05, 3)]
        self.velocities = array(velocity[None], wp.spatial_vector)
        nodes = np.zeros((capacity, 4), np.int32)
        bary = np.zeros((capacity, 4), np.float32)
        bary[:, 0] = 1
        for row in range(count):
            nodes[row] = rng.choice(np.arange(1, self.PARTICLES), 4, replace=False)
            bary[row] = rng.dirichlet(np.ones(4))
        self.nodes = wp.array(nodes, dtype=wp.vec4i, device=DEVICE)
        self.bary = array(bary, wp.vec4)
        lengths, engaged = np.ones(capacity, np.float32), np.ones(capacity, np.int32)
        lengths[:count] = rng.uniform(1.0e-4, 3.0e-4, count)
        # One released sample and one without a quadrature length among the samples.
        engaged[1], lengths[2] = 0, 0.0
        self.lengths = array(lengths, float)
        self.engaged = wp.array(engaged, dtype=int, device=DEVICE)
        live = np.zeros(3, np.int32)
        live[LIVE_TRACK] = count
        self.live = wp.array(live, dtype=int, device=DEVICE)
        normal = np.full((capacity, 3), np.nan, np.float32)
        normal[:count] = rng.normal(0, 0.3, (count, 3))
        drag = np.full(capacity, np.nan, np.float32)
        # Held drag multipliers at their bound, both ways, and none: sliding and sticking samples.
        drag[:count] = rng.choice([-12.0, 0.0, 12.0], count)
        self.normal, self.drag = array(normal, wp.vec3), array(drag, float)
        self.h = array([H], float)
        fixed = np.zeros(self.PARTICLES, np.int32)
        fixed[rng.choice(np.arange(1, self.PARTICLES), 3, replace=False)] = 1
        self.fixed = wp.array(fixed, dtype=int, device=DEVICE)
        n = self.PARTICLES + 2
        x = rng.normal(0, 0.1, (n, 3))
        x[0] = [np.inf, np.nan, -np.inf]
        self.x = array(x, wp.vec3)
        # Anchors as an anchor kernel leaves them for the samples; NaN beyond.
        self.matrices = array(self.padded(rng.normal(0, 0.2, (count, 3, 3)) @ np.diag([1.0, 0.5, 0.0])), wp.mat33)
        self.levers = array(self.padded(rng.normal(0, 0.5, (count, 3))), wp.vec3)
        self.shears = array(self.padded(rng.normal(0, 0.1, (count, 3, 3))), wp.mat33)
        self.out = rng.normal(0, 0.1, (n, 3)).astype(np.float32)
        self.diagonal = rng.normal(0, 0.1, (n, 3, 3)).astype(np.float32)
        self.output = rng.normal(0, 0.1, 3 * n)
        projected = wp.full((capacity, 8), np.nan, dtype=float, device=DEVICE)
        wp.launch(
            original_packet,
            count,
            [self.nodes, self.bary, self.q, self.poses, 0, self.RADIUS, self.samples, projected],
            device=DEVICE,
        )
        self.projected = projected

    def padded(self, rows):
        """``rows`` of the samples in storage of the reserve, NaN in every row beyond them."""
        values = np.full((self.capacity, *rows.shape[1:]), np.nan, np.float32)
        values[: self.count] = rows
        return values

    def anchor_inputs(self, alm, compliance):
        """Arguments of an anchor kernel without its bound, up to its outputs."""
        head = [self.nodes, self.bary, self.lengths, self.engaged, self.q, self.qd, self.samples, self.poses]
        head += [self.velocities, 0, wp.vec3(0.0), self.RADIUS]
        if alm:
            return [*head, self.normal, self.drag, self.PENALTY, compliance, self.CLEARANCE, 0.2, self.h, 0.01]
        return [*head, 6000.0, 0.4, 0.2, 1.0e-3, self.h, 0.01]


def with_bound(inputs, position, live):
    """Arguments of the kernel with its bound: the live counts at ``position``."""
    return [*inputs[:position], live, *inputs[position:]]


class BoundedKernelTests(unittest.TestCase):
    """Seven kernels, each over the reserve against its text without the bound over the samples."""

    COUNT, CAPACITY = 6, 11

    def compare(self, name, track, inputs, position, outputs):
        """Launch ``name`` over the reserve and its text without the bound over the samples, from the same values.

        ``outputs`` maps each output parameter to its start values. Returns the arrays that the
        bounded kernel wrote.
        """
        kernel, earlier = getattr(_MODULES[name], name), unbounded(name)
        results = []
        for bounded, dim in ((True, track.capacity), (False, track.count), (False, track.capacity)):
            arrays = {key: wp.array(value, dtype=dtype, device=DEVICE) for key, (value, dtype) in outputs.items()}
            arguments = [*inputs, *arrays.values()]
            if bounded:
                arguments = with_bound(arguments, position, track.live)
            wp.launch(kernel if bounded else earlier, dim, arguments, device=DEVICE)
            results.append({key: value.numpy().copy() for key, value in arrays.items()})
        over_reserve, over_samples, unbounded_over_reserve = results
        for key, (start, _) in outputs.items():
            written = over_reserve[key]
            if DEVICE.is_cuda and name in ("contact_action", "contact_diagonal", "balanced_correction"):
                # Sums over the samples have no fixed order on CUDA.
                scale = np.abs(over_samples[key][np.isfinite(over_samples[key])]).max()
                np.testing.assert_allclose(written, over_samples[key], rtol=0, atol=2.0e-6 * scale, err_msg=key)
            else:
                self.assertEqual(written.tobytes(), over_samples[key].tobytes(), f"{name}: {key}")
            self.assertNotEqual(written.tobytes(), np.asarray(start).tobytes(), f"{name}: {key} was not written")
            self.assertTrue(np.isfinite(written[np.isfinite(start)]).all(), f"{name}: {key}")
            # The same text without the bound over the reserve runs the rows beyond the samples,
            # which poison the result: the comparison above can fail.
            self.assertNotEqual(unbounded_over_reserve[key].tobytes(), written.tobytes(), f"{name}: {key} control")
        return over_reserve

    def test_anchor_kernels_write_the_rows_of_the_samples_and_no_other(self):
        track = Track(7, self.COUNT, self.CAPACITY)
        sentinel = {
            "matrices": (np.full((track.capacity, 3, 3), -7.25, np.float32), wp.mat33),
            "levers": (np.full((track.capacity, 3), -7.25, np.float32), wp.vec3),
            "shears": (np.full((track.capacity, 3, 3), -7.25, np.float32), wp.mat33),
        }
        penalty = {key: sentinel[key] for key in ("matrices", "levers")}
        cases = [("prepare_anchors", False, 0.0, penalty, None)]
        # Compliant wall (the grasp scene) and hard wall.
        cases += [("prepare_alm_anchors", True, compliance, sentinel, None) for compliance in (1.0 / 300000.0, 0.0)]
        cases += [("prepare_anchors_projected", False, 0.0, penalty, track.projected)]
        cases += [("prepare_alm_anchors_projected", True, 1.0 / 300000.0, sentinel, track.projected)]
        for name, alm, compliance, outputs, projected in cases:
            with self.subTest(kernel=name, compliance=compliance):
                inputs = track.anchor_inputs(alm, compliance)
                # The projection of a projected twin follows the outputs: hand it over as one more
                # "output" that no kernel writes.
                written = self.compare_anchor(name, track, inputs, outputs, projected)
                for key, values in written.items():
                    # Rows of the samples are written (a released one with zeros), the others keep their start.
                    self.assertFalse(np.any(values[: track.count] == -7.25), key)
                    np.testing.assert_array_equal(values[track.count :], -7.25, err_msg=key)
                np.testing.assert_array_equal(written["matrices"][1], 0)
                np.testing.assert_array_equal(written["matrices"][2], 0)
                self.assertTrue(np.any(written["matrices"][0]))
                if alm:
                    # The fixture has sliding samples: their shear is written.
                    self.assertTrue(np.any(written["shears"][: track.count]))

    def compare_anchor(self, name, track, inputs, outputs, projected):
        kernel, earlier = getattr(_MODULES[name], name), unbounded(name)
        results = []
        for bounded, dim in ((True, track.capacity), (False, track.count), (False, track.capacity)):
            arrays = {key: wp.array(value, dtype=dtype, device=DEVICE) for key, (value, dtype) in outputs.items()}
            arguments = [*inputs, *arrays.values()]
            if projected is not None:
                arguments.append(projected)
            if bounded:
                # The live counts follow the engagement.
                arguments = with_bound(arguments, 4, track.live)
            wp.launch(kernel if bounded else earlier, dim, arguments, device=DEVICE)
            results.append({key: value.numpy().copy() for key, value in arrays.items()})
        over_reserve, over_samples, unbounded_over_reserve = results
        for key in outputs:
            # One thread writes one row: the same bits on every device.
            self.assertEqual(over_reserve[key].tobytes(), over_samples[key].tobytes(), f"{name}: {key}")
        # Without the bound the rows beyond the samples are written, from values that are not finite.
        self.assertFalse(np.any(unbounded_over_reserve["matrices"][track.count :] == -7.25), name)
        self.assertFalse(np.isfinite(unbounded_over_reserve["levers"][track.count :]).any(), name)
        return over_reserve

    def test_kernels_that_apply_the_anchors_add_nothing_for_a_row_beyond_the_samples(self):
        track = Track(11, self.COUNT, self.CAPACITY)
        head = [track.fixed, track.nodes, track.bary, track.matrices, track.levers, track.shears, track.h]
        # The live counts follow the weights.
        self.compare("contact_action", track, [*head, track.x], 3, {"out": (track.out, wp.vec3)})
        self.compare("contact_diagonal", track, head, 3, {"diagonal": (track.diagonal, wp.mat33)})
        self.compare("balanced_correction", track, [*head, track.x], 3, {"output": (track.output, wp.float64)})

    def test_no_track_sample_writes_nothing(self):
        track = Track(13, self.COUNT, self.CAPACITY)
        track.live.zero_()
        head = [track.fixed, track.nodes, track.bary, track.live, track.matrices, track.levers, track.shears, track.h]
        for kernel, inputs, start, dtype in (
            (shaft_preconditioner.contact_action, [*head, track.x], track.out, wp.vec3),
            (shaft_preconditioner.contact_diagonal, head, track.diagonal, wp.mat33),
            (shaft_preconditioner.balanced_correction, [*head, track.x], track.output, wp.float64),
        ):
            with self.subTest(kernel=kernel.key):
                output = wp.array(start, dtype=dtype, device=DEVICE)
                wp.launch(kernel, track.capacity, [*inputs, output], device=DEVICE)
                self.assertEqual(output.numpy().tobytes(), np.asarray(start).tobytes())


# ---------------------------------------------------------------------------------------------
# The correction of an embedded shaft in storage of different sizes.


def embedded(stack, *, capacity=None, law="penalty", prepare=setup, engage_last=True):
    """The embedded shaft of the preconditioner tests; ``capacity`` rows of track storage (None: the example's reserve).

    The probes leave their newest sample released. ``engage_last`` engages it, so that the last
    row of the track carries an anchor and a launch that stops one row early changes a result.
    Returns the example, the correction that was made with the interaction, before the first
    track sample, the storage warnings of the track that the probes laid, and what ``prepare`` returned.
    """
    example = stack.enter_context(_example())
    reserve, start, made = example._history_reserve, example.start, []

    def reserved():
        return reserve() if capacity is None else dict(reserve(), track_capacity=capacity)

    def started():
        start()
        made.append(_ShaftPreconditioner(example.coupler))

    with (
        patch.object(example, "_history_reserve", side_effect=reserved),
        patch.object(example, "start", side_effect=started),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = prepare(example, H, 6000)
    interaction = example.interaction
    if engage_last:
        engaged = wp.array([1], dtype=int, device="cpu")
        wp.copy(interaction._track_engaged, engaged, dest_offset=interaction.track_count - 1, count=1)
        # The engagement was written in place: an engaged length that was read before is not its.
        interaction._engaged_length = None
    if law == "alm":
        # A friction tunnel on the track that the probes laid, as a first step would size it.
        interaction.friction, interaction.shaft_damping_per_m = 0.02, 0
        interaction.tunnel_alm = _TunnelALM(
            interaction, penalty_per_m=6000, compliance_per_m=1 / 6000, clearance=2e-5, tolerance=2e-6
        )
        interaction.tunnel_alm._resize()
        stack.callback(interaction.tunnel_alm.discard)
    growths = [str(warning.message) for warning in caught if "track storage grew" in str(warning.message)]
    return example, made[0], growths, result


def correct(example, pre, scales=(1.0, -2.0)):
    """Corrections of ``pre`` for one residual at several scales: the outputs and what the solve left behind."""
    interaction = example.interaction
    if interaction.tunnel_alm is not None:
        interaction.tunnel_alm.begin_step()
        # Drag multipliers beyond the Coulomb bound of the wall reactions: the samples slide and
        # their anchors carry a shear, so the system of the solve is not symmetric.
        interaction.tunnel_alm.tangent.fill_(12.0)
    pre.begin_step()
    count = interaction.track_count
    rng = np.random.default_rng(831)
    values = rng.normal(0, 1e-5, 3 * pre.n)
    result = {}
    for scale in scales:
        residual = wp.array(scale * values, dtype=wp.float64, device=example.device)
        result[f"output {scale}"] = _apply(pre, example.state_0, H, residual).numpy().copy()
        for name in ("z", "rhs", "diagonal", "inverse_diagonal"):
            result[f"{name} {scale}"] = getattr(pre, name).numpy().copy()
        for name in ("matrices", "levers", "shears"):
            result[f"{name} {scale}"] = getattr(pre, name).numpy()[:count].copy()
    if interaction.tunnel_alm is not None:
        interaction.tunnel_alm.discard()
    return result, values


def assert_same(test, actual, expected, *, exact):
    test.assertEqual(actual.keys(), expected.keys())
    for name, values in expected.items():
        if exact or name.split()[0] in ("matrices", "levers", "shears", "rhs"):
            test.assertEqual(actual[name].tobytes(), values.tobytes(), name)
        else:
            # Sums over samples have no fixed order on CUDA.
            np.testing.assert_allclose(actual[name], values, rtol=0, atol=1e-4 * np.abs(values).max(), err_msg=name)


class CorrectionStorageTests(unittest.TestCase):
    def test_correction_in_exact_and_in_padded_storage_holds_the_same_bits(self):
        for law in ("penalty", "alm"):
            with self.subTest(law=law), ExitStack() as stack:
                padded, _, growths, _ = embedded(stack, law=law)
                count = padded.interaction.track_count
                self.assertGreater(count, 2)
                self.assertGreater(padded.interaction.track_capacity, count + 1)
                self.assertEqual(growths, [])
                exact, _, growths, _ = embedded(stack, law=law, capacity=count)
                self.assertEqual((exact.interaction.track_count, exact.interaction.track_capacity), (count, count))
                self.assertEqual(growths, [])
                results = []
                for example in (exact, padded):
                    pre = _ShaftPreconditioner(example.coupler)
                    interaction = example.interaction
                    self.assertEqual((pre.capacity, pre.count), (interaction.track_capacity, count))
                    for name, dtype in (("matrices", wp.mat33), ("levers", wp.vec3), ("shears", wp.mat33)):
                        anchors = getattr(pre, name)
                        self.assertEqual((anchors.shape, anchors.dtype), ((pre.capacity,), dtype), name)
                        np.testing.assert_array_equal(anchors.numpy(), 0, err_msg=name)
                        if pre.capacity > count:
                            # Rows beyond the samples that would poison a correction that read them.
                            poison = anchors.numpy()
                            poison[count:] = np.nan
                            anchors.assign(poison)
                    result, values = correct(example, pre)
                    self.assertGreater(np.abs(result["output 1.0"] - values).max(), 0)
                    for name in ("matrices", "levers", "shears"):
                        # No kernel wrote a row beyond the samples.
                        self.assertTrue(np.isnan(getattr(pre, name).numpy()[count:]).all(), name)
                    self.assertEqual(pre.storage_growths, 0)
                    self.assertEqual(pre.nonsymmetric, law == "alm")
                    self.assertIs(pre.linear_solver, pre._sheared_solver if law == "alm" else pre._symmetric_solver)
                    results.append(result)
                # Most samples are loaded, the last row of the track among them, and the samples of
                # the friction tunnel slide.
                loaded = np.any(results[0]["matrices 1.0"], axis=(1, 2))
                self.assertTrue(loaded[-1])
                self.assertGreater(int(loaded.sum()), count // 2)
                self.assertEqual(bool(np.any(results[0]["shears 1.0"][-1])), law == "alm")
                assert_same(self, results[1], results[0], exact=not DEVICE.is_cuda)

    @unittest.skipIf(DEVICE.is_cuda, "bitwise on the CPU: the native host path, as the consumer runs it")
    def test_native_host_step_in_exact_and_in_padded_storage_accepts_the_same_bits(self):
        records = []
        with ExitStack() as stack:
            # The track as the probes left it: this is an accepted physical step.
            options = dict(prepare=lambda ex, *_: native_embedded(ex), engage_last=False)
            padded, _, _, padded_native = embedded(stack, **options)
            count = padded.interaction.track_count
            self.assertGreater(padded.interaction.track_capacity, count)
            exact, _, _, exact_native = embedded(stack, capacity=count, **options)
            self.assertEqual((exact.interaction.track_count, exact.interaction.track_capacity), (count, count))
            for example, (h, _native, coupler) in ((exact, exact_native), (padded, padded_native)):
                pre = coupler._shaft_preconditioner
                self.assertEqual(pre.capacity, example.interaction.track_capacity)
                coupler.step(example.state_0, example.state_1, example.control, h)
                self.assertTrue(pre.active)
                record = {
                    name: getattr(example.state_1, name).numpy().copy()
                    for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
                }
                record.update(load=coupler._load.numpy().copy(), value=coupler._value.numpy().copy())
                record.update(z=pre.z.numpy().copy(), direction=coupler._direction.numpy().copy())
                record.update(trials=np.array([coupler.iterations]), residual=np.array([coupler.residual]))
                record.update(engaged=example.interaction._track_engaged.numpy()[:count].copy())
                records.append(record)
        self.assertGreater(records[0]["trials"][0], 1)
        self.assertGreater(np.abs(records[0]["z"]).max(), 0)
        for name, values in records[0].items():
            self.assertEqual(records[1][name].tobytes(), values.tobytes(), name)

    def test_reserve_that_was_too_small_is_followed_once_and_gives_the_bits_of_a_large_one(self):
        with ExitStack() as stack:
            large, _, _, _ = embedded(stack)
            count = large.interaction.track_count
            expected, _ = correct(large, _ShaftPreconditioner(large.coupler))
            small, pre, growths, _ = embedded(stack, capacity=2)
            interaction = small.interaction
            self.assertEqual(interaction.track_count, count)
            # The interaction doubled its storage for the track that the probes laid, and warned.
            self.assertGreaterEqual(interaction.track_capacity, count)
            self.assertEqual(len(growths), interaction.track_storage_growths)
            self.assertGreaterEqual(interaction.track_storage_growths, 1)
            # The correction was made with the interaction, at the reserve of two rows.
            anchors = (pre.matrices, pre.levers, pre.shears)
            self.assertEqual((pre.capacity, pre.count, pre.storage_growths), (2, 0, 0))
            key = pre.storage_key()
            self.assertEqual(key[0], 2)
            result, _ = correct(small, pre)
            # Its first staging after the growth made the anchors anew, once, whatever the number of doublings.
            self.assertEqual((pre.capacity, pre.count, pre.storage_growths), (interaction.track_capacity, count, 1))
            for before, name in zip(anchors, ("matrices", "levers", "shears"), strict=True):
                self.assertIsNot(getattr(pre, name), before, name)
                self.assertEqual(getattr(pre, name).shape, (interaction.track_capacity,), name)
            self.assertNotEqual(pre.storage_key(), key)
            self.assertEqual(pre.storage_key()[0], interaction.track_capacity)
            assert_same(self, result, expected, exact=not DEVICE.is_cuda)
            # Nothing more is made while the storage stays.
            anchors, key = (pre.matrices, pre.levers, pre.shears), pre.storage_key()
            again, _ = correct(small, pre)
            assert_same(self, again, expected, exact=not DEVICE.is_cuda)
            self.assertEqual(pre.storage_growths, 1)
            self.assertEqual(pre.storage_key(), key)
            for before, name in zip(anchors, ("matrices", "levers", "shears"), strict=True):
                self.assertIs(getattr(pre, name), before, name)


# ---------------------------------------------------------------------------------------------
# Staging: what the host writes before a correction.


def damping_writes(pre):
    """Count the uploads of the native damping of ``pre`` while the returned patch is active."""
    writes, assign = [], wp.array.assign

    def counted(array, *args, **kwargs):
        if array is pre.native_damping:
            writes.append(np.asarray(args[0]).copy())
        return assign(array, *args, **kwargs)

    return writes, patch.object(wp.array, "assign", counted)


class StagingTests(unittest.TestCase):
    def test_staging_writes_what_changed_and_refuses_without_writing(self):
        with ExitStack() as stack:
            example, _, _, _ = embedded(stack)
            pre = _ShaftPreconditioner(example.coupler)
            interaction, native = example.interaction, example.rigid_solver.mj_model
            count = interaction.track_count
            anchors = (pre.matrices, pre.levers, pre.shears)
            native.dof_damping[:6] = [0.002, 0.003, 0.004, 2e-7, 3e-7, 5e-7]
            writes, counting = damping_writes(pre)
            with counting:
                self.assertTrue(pre.stage(H, tape=True))
                self.assertEqual(len(writes), 1)
                np.testing.assert_array_equal(pre.native_damping.numpy(), native.dof_damping[:6].astype(np.float32))
                np.testing.assert_array_equal(pre.time_step.numpy(), [np.float32(H)])
                step_writes = pre.step_time.writes
                # The same values again: nothing is uploaded and nothing is made.
                for tape in (True, False, True):
                    self.assertTrue(pre.stage(H, tape=tape))
                self.assertEqual((len(writes), pre.step_time.writes), (1, step_writes))
                self.assertEqual((pre.count, pre.capacity, pre.storage_growths), (count, interaction.track_capacity, 0))
                for before, name in zip(anchors, ("matrices", "levers", "shears"), strict=True):
                    self.assertIs(getattr(pre, name), before, name)
                # A changed time step and a changed damping are each written once.
                self.assertTrue(pre.stage(H / 2, tape=False))
                self.assertEqual((len(writes), pre.step_time.writes), (1, step_writes + 1))
                native.dof_damping[2] = 0.005
                self.assertTrue(pre.stage(H / 2, tape=False))
                self.assertEqual((len(writes), pre.step_time.writes), (2, step_writes + 1))
                np.testing.assert_array_equal(pre.native_damping.numpy(), native.dof_damping[:6].astype(np.float32))
                # An invalid damping is refused before anything is written, with or without the tape test.
                staged = pre.native_damping.numpy().copy()
                for value in (-0.1, np.nan, np.inf):
                    native.dof_damping[1] = value
                    for tape in (True, False):
                        self.assertFalse(pre.stage(H, tape=tape))
                self.assertEqual((len(writes), pre.step_time.writes), (2, step_writes + 1))
                self.assertEqual(pre.step_time.value, H / 2)
                np.testing.assert_array_equal(pre.native_damping.numpy(), staged)
                native.dof_damping[1] = 0.003
                # Under a tape the correction of a track is refused; the staging of an empty track is not.
                with wp.Tape():
                    self.assertFalse(pre.stage(H, tape=True))
                    self.assertEqual(pre.step_time.value, H / 2)
                    self.assertTrue(pre.stage(H, tape=False))
                self.assertEqual(pre.step_time.value, H)
                for value in (0.0, -H, np.nan, np.inf):
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        pre.stage(value, tape=False)
            # prepare is the staging behind the guard of an inactive, empty or lawless track.
            pre.begin_step()
            self.assertTrue(pre.active)
            self.assertTrue(pre.prepare(H))
            stiffness, damping = interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m
            interaction.shaft_stiffness_per_m = interaction.shaft_damping_per_m = 0.0
            self.assertFalse(pre.prepare(H))
            interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m = stiffness, damping
            pre.active = False
            self.assertFalse(pre.prepare(H))
            pre.active = True
            with patch.object(type(interaction), "track_count", new_callable=PropertyMock, return_value=0):
                self.assertFalse(pre.prepare(H))
            self.assertEqual(pre.count, count)

    @unittest.skipUnless(DEVICE.is_cuda, "Captured articulated trial loop")
    def test_articulated_trials_stage_through_the_correction_and_keep_their_refusals(self):
        import test_mjw_device_trials as articulated

        with articulated._pair() as (_, scene):
            coupler, interaction = scene.coupler, scene.interaction
            pre, native = coupler._shaft_preconditioner, scene.rigid_solver.mj_model
            scene.coupler.step(
                scene.state_0, scene.state_1, scene.control, articulated.DT, external_forces=scene._external_forces
            )
            self.assertEqual(coupler.coupling_graph_fallbacks, 0)
            rigid = coupler._captured_trials.rigid
            count = interaction.track_count
            self.assertGreater(count, 1)
            # The start guess of the correction was bound in that step; its key stays from then on.
            guess_key = rigid.guess_key
            self.assertIsNotNone(guess_key)
            anchors = (pre.matrices, pre.levers, pre.shears)
            # The trial graph keeps the anchors alive that it addresses.
            for array in anchors:
                self.assertTrue(any(owner is array for owner in rigid.capture_owners))
            self.assertEqual((pre.count, pre.capacity, pre.storage_growths), (count, interaction.track_capacity, 0))
            writes, counting = damping_writes(pre)
            with counting:
                step_writes = pre.step_time.writes
                # A preparation with nothing changed uploads nothing, makes nothing and leaves the
                # correction's own activity flag alone.
                for active in (True, False):
                    pre.active = active
                    rigid.prepare(articulated.DT)
                    rigid.prepare(articulated.DT)
                    self.assertEqual(pre.active, active)
                pre.active = True
                self.assertEqual((len(writes), pre.step_time.writes), (0, step_writes))
                for before, name in zip(anchors, ("matrices", "levers", "shears"), strict=True):
                    self.assertIs(getattr(pre, name), before, name)
                # With a track: an invalid damping and a track without a shaft law refuse the captured
                # operator. An active tape is refused before that, by the rigid checkpoint.
                kept = float(native.dof_damping[1])
                native.dof_damping[1] = -0.1
                with self.assertRaisesRegex(ValueError, "Unsupported captured contact operator"):
                    rigid.prepare(articulated.DT)
                native.dof_damping[1] = kept
                with wp.Tape(), self.assertRaisesRegex(ValueError, "do not implement automatic derivatives"):
                    rigid.prepare(articulated.DT)
                alm, stiffness, damping = (
                    interaction.tunnel_alm,
                    interaction.shaft_stiffness_per_m,
                    interaction.shaft_damping_per_m,
                )
                interaction.tunnel_alm = None
                interaction.shaft_stiffness_per_m = interaction.shaft_damping_per_m = 0.0
                try:
                    with self.assertRaisesRegex(ValueError, "Unsupported captured contact operator"):
                        rigid.prepare(articulated.DT)
                finally:
                    interaction.tunnel_alm = alm
                    interaction.shaft_stiffness_per_m, interaction.shaft_damping_per_m = stiffness, damping
                # Without a track the surface-only system is prepared: its one refusal is the damping.
                with patch.object(type(interaction), "track_count", new_callable=PropertyMock, return_value=0):
                    native.dof_damping[1] = np.nan
                    with self.assertRaisesRegex(ValueError, "Invalid rigid damping"):
                        rigid.prepare(articulated.DT)
                    native.dof_damping[1] = kept
                    rigid.prepare(articulated.DT)
                    self.assertEqual(pre.count, 0)
                    # No track sample: the guess keeps its key, and its device flag says that
                    # this launch may not use it.
                    self.assertEqual(rigid.guess_key, guess_key)
                    self.assertEqual(rigid.guess.available.numpy().tolist(), [0])
                self.assertEqual(len(writes), 0)
            rigid.prepare(articulated.DT)
            self.assertEqual(pre.count, count)
            for before, name in zip(anchors, ("matrices", "levers", "shears"), strict=True):
                self.assertIs(getattr(pre, name), before, name)
            # The next step replays what was captured.
            captures = coupler.coupling_graph_captures
            scene.state_0, scene.state_1 = scene.state_1, scene.state_0
            scene.coupler.step(
                scene.state_0, scene.state_1, scene.control, articulated.DT, external_forces=scene._external_forces
            )
            self.assertEqual(coupler.coupling_graph_fallbacks, 0)
            self.assertLessEqual(coupler.coupling_graph_captures, captures + 1)


# ---------------------------------------------------------------------------------------------
# Graph keys: the storage, not the track count.


def append_released_sample(example):
    """One more track sample far from the needle, without an edge to the track and released.

    It has no quadrature length and exerts no force: only the history grows.
    """
    x = example.interaction
    count = x.track_count
    centers = x._previous_q[x.tetrahedra].mean(axis=1)
    order = np.argsort(-np.linalg.norm(centers - x._previous_tip, axis=1))
    # A tetrahedron that the needle did not visit and that holds no track sample yet: a second
    # sample at the place of an earlier one would not be recorded.
    sampled = {tuple(int(node) for node in nodes) for nodes in x._track_nodes_host}
    tet = int(
        next(
            tet
            for tet in order
            if int(tet) not in x._visited_tets and tuple(int(node) for node in x.tetrahedra[tet]) not in sampled
        )
    )
    anchor, x._track_anchor = x._track_anchor, None
    x._record_track(x.tetrahedra[tet].copy(), np.full(4, 0.25))
    x._track_anchor = anchor
    if x.track_count != count + 1:
        raise AssertionError("The sample was not recorded")
    released = wp.array([0], dtype=int, device="cpu")
    wp.copy(x._track_engaged, released, dest_offset=count, count=1)
    x._engaged_length = None
    return x.track_count


class StorageKeyTests(unittest.TestCase):
    def test_storage_key_holds_the_storage_and_not_the_track_count(self):
        with ExitStack() as stack:
            example, _, _, _ = embedded(stack)
            pre = _ShaftPreconditioner(example.coupler)
            interaction = example.interaction
            key = pre.storage_key()
            self.assertEqual(key[0], interaction.track_capacity)
            first, _ = correct(example, pre)
            self.assertEqual(pre.storage_key(), key)
            # A new track sample: the count follows, the storage and its key stay.
            anchors = (pre.matrices, pre.levers, pre.shears)
            count = append_released_sample(example)
            self.assertLess(count, interaction.track_capacity)
            second, _ = correct(example, pre)
            self.assertEqual((pre.count, pre.capacity, pre.storage_growths), (count, interaction.track_capacity, 0))
            self.assertEqual(pre.storage_key(), key)
            for before, name in zip(anchors, ("matrices", "levers", "shears"), strict=True):
                self.assertIs(getattr(pre, name), before, name)
            # The released sample has no anchor: the correction is the one before it.
            np.testing.assert_array_equal(pre.matrices.numpy()[count - 1], 0)
            self.assertEqual(second["output 1.0"].tobytes(), first["output 1.0"].tobytes())
            # Every array that a captured correction addresses on this object is in the key.
            names = ["matrices", "levers", "shears", "time_step", "native_damping"]
            for name in names:
                kept = getattr(pre, name)
                setattr(pre, name, wp.clone(kept))
                self.assertNotEqual(pre.storage_key(), key, name)
                setattr(pre, name, kept)
                self.assertEqual(pre.storage_key(), key, name)
            # And the values of the projected tissue operator, where the tissue solver holds one: the
            # correction applies it for the tissue rows.
            fixed = example.solver._fixed_particles
            kept = fixed._projected
            values = wp.zeros(4, dtype=wp.mat33, device=example.device)
            fixed._projected = SimpleNamespace(values=values)
            try:
                with_projection = pre.storage_key()
                self.assertNotEqual(with_projection, key)
                fixed._projected.values = wp.clone(values)
                self.assertNotEqual(pre.storage_key(), with_projection)
            finally:
                fixed._projected = kept
            self.assertEqual(pre.storage_key(), key)

    @unittest.skipUnless(DEVICE.is_cuda, "CUDA graph replay contract")
    def test_new_track_sample_replays_the_solve_and_preparation_graphs_and_replaced_storage_captures_again(self):
        with ExitStack() as stack:
            example, _, _, _ = embedded(stack)
            reference, _, _, _ = embedded(stack)
            pre, eager = _ShaftPreconditioner(example.coupler), _ShaftPreconditioner(reference.coupler)
            eager.capture_preparation = False
            values = np.random.default_rng(57).normal(0, 1e-5, 3 * pre.n)
            residual = wp.array(values, dtype=wp.float64, device=example.device)
            other = wp.array(values, dtype=wp.float64, device=reference.device)
            output, expected = wp.empty_like(residual), wp.empty_like(other)

            def apply(scale):
                residual.assign(scale * values)
                other.assign(scale * values)
                pre.apply(example.state_0, H, residual, output)
                # The reference issues the same application as written: no graph of its own.
                eager._graph_key = None
                eager.apply(reference.state_0, H, other, expected)
                self.assertEqual((eager.graph_captures, eager.graph_launches), (0, 0))
                correction = np.abs(expected.numpy() - other.numpy()).max()
                self.assertGreater(correction, 0)
                np.testing.assert_allclose(output.numpy(), expected.numpy(), rtol=0, atol=1e-4 * correction)

            def counters():
                replay = pre._input_preparation.replay
                return (pre.graph_captures, pre.graph_launches, replay.eager, replay.captures, replay.launches)

            # As written, captured, launched.
            for scale in (1.0, -2.0, 0.5):
                apply(scale)
            self.assertEqual(counters(), (1, 2, 1, 1, 2))
            graph, key = pre._graph, pre._graph_key
            preparation = pre._input_preparation.replay
            prepared, prepared_key = preparation.graph, preparation.key
            # New track samples: both graphs are launched again, with the count that the device holds.
            # The first is released and has no anchor. The second is engaged and has a length: a graph
            # that kept the count of its capture would leave its anchor out and differ from the
            # application as written on the twin.
            for index, append in enumerate((append_released_sample, append_engaged_sample)):
                for target in (example, reference):
                    count = append(target)
                apply(1.5 + index)
                self.assertEqual(pre.count, count)
                self.assertEqual(counters(), (1, 3 + index, 1, 1, 3 + index))
                self.assertIs(pre._graph, graph)
                self.assertEqual(pre._graph_key, key)
                self.assertIs(preparation.graph, prepared)
                self.assertEqual(preparation.key, prepared_key)
                # The launched preparation wrote the anchor of the new row, or left the released one empty.
                self.assertEqual(bool(np.any(pre.matrices.numpy()[count - 1])), append is append_engaged_sample)
            np.testing.assert_array_equal(example.interaction._live.numpy()[LIVE_TRACK], count)
            # Replaced storage that the solve addresses: as written once more, then captured.
            for step, name in enumerate(("time_step", "native_damping", "levers")):
                setattr(pre, name, wp.clone(getattr(pre, name)))
                self.assertNotEqual(pre.storage_key(), key[0], name)
                apply(0.7)
                apply(-0.3)
                self.assertEqual(pre.graph_captures, 2 + step, name)
                self.assertIsNot(pre._graph, graph, name)
                self.assertNotEqual(pre._graph_key, key, name)
                self.assertEqual(preparation.captures, 2 + step, name)
                graph, key = pre._graph, pre._graph_key

    @unittest.skipUnless(DEVICE.is_cuda, "Captured trial loop")
    def test_replaced_storage_of_the_correction_captures_the_trial_graphs_again(self):
        with coupled._pair("penalty") as (reference, captured):
            coupler = captured.coupler
            pre = coupler._shaft_preconditioner

            def step():
                for example in (reference, captured):
                    example.coupler.step(example.state_0, example.state_1, example.control, coupled._DT)
                    example.state_0, example.state_1 = example.state_1, example.state_0
                self.assertEqual(coupler.coupling_graph_fallbacks, 0)

            step()
            step()
            self.assertEqual(coupler.coupling_graph_captures, 2)
            signature = coupler._captured_trials.signature
            self.assertIn(pre.storage_key(), signature)
            entry = signature.index(pre.storage_key())
            captures = 2
            for name in ("time_step", "native_damping", "matrices"):
                with self.subTest(replaced=name):
                    setattr(pre, name, wp.clone(getattr(pre, name)))
                    step()
                    step()
                    captures += 2
                    current = coupler._captured_trials.signature
                    changed = [i for i, (old, new) in enumerate(zip(signature, current, strict=True)) if old != new]
                    self.assertEqual(changed, [entry])
                    self.assertEqual(current[entry], pre.storage_key())
                    self.assertEqual(coupler.coupling_graph_captures, captures)
                    signature = current
                    for field in ("particle_q", "body_q"):
                        np.testing.assert_allclose(
                            getattr(captured.state_0, field).numpy(),
                            getattr(reference.state_0, field).numpy(),
                            rtol=3e-6,
                            atol=5e-8,
                            err_msg=field,
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
