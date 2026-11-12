# SPDX-License-Identifier: Apache-2.0
"""Trial preparation captures all state/history copies without stale buffers."""

import os
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp

from newton_solver_suturebot._src.coupling import _apply_interface_load
from newton_solver_suturebot._src.host_trial_prepare import CapturedTrialPreparation, _arrays


def state(device):
    result = newton.State()
    for name in ("particle_q", "particle_qd", "particle_f"):
        setattr(result, name, wp.zeros(3, dtype=wp.vec3, device=device))
    result.body_q = wp.zeros(2, dtype=wp.transform, device=device)
    result.body_qd = wp.zeros(2, dtype=wp.spatial_vector, device=device)
    result.body_f = wp.zeros(2, dtype=wp.spatial_vector, device=device)
    result.joint_q = wp.zeros(4, dtype=float, device=device)
    result.joint_qd = wp.zeros(3, dtype=float, device=device)
    result._tendon_initialized = wp.zeros(2, dtype=wp.bool, device=device)
    result.history = newton.Model.AttributeNamespace("history")
    result.history.previous = wp.zeros(3, dtype=wp.float64, device=device)
    return result


def coupler(device):
    return SimpleNamespace(
        model=SimpleNamespace(device=device, particle_count=3, body_count=2, requires_grad=False),
        capture_fem=True,
        needle_body=1,
        moment_length=0.01,
        _candidate=state(device),
        _trial_load=wp.zeros(15, dtype=wp.float64, device=device),
        _external_nodes=wp.zeros(3, dtype=wp.vec3, device=device),
        _external_wrench=wp.zeros(2, dtype=wp.spatial_vector, device=device),
        _forces=wp.zeros(3, dtype=wp.vec3, device=device),
    )


def eager(c, incoming):
    wp.launch(
        _apply_interface_load,
        max(c.model.particle_count, c.model.body_count),
        [
            c._trial_load,
            c._external_nodes,
            c._external_wrench,
            c.needle_body,
            c.moment_length,
            c._forces,
            incoming.body_f,
        ],
        device=c.model.device,
    )
    c._candidate.assign(incoming)


class HostTrialPrepareTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))

    def test_alternating_states_and_mutable_loads_match_every_eager_field(self):
        reference, captured = coupler(self.device), coupler(self.device)
        actual = [state(self.device), state(self.device)]
        expected = [state(self.device), state(self.device)]
        helper = CapturedTrialPreparation(captured)
        for trial in range(12):
            current, baseline = actual[trial % 2], expected[trial % 2]
            for index, ((name, array), (_, other)) in enumerate(zip(_arrays(current), _arrays(baseline), strict=True)):
                array.fill_((trial + index) % 2 if name == "_tendon_initialized" else trial + index + 0.25)
                other.assign(array)
            for c in (reference, captured):
                c._trial_load.assign(np.arange(15, dtype=float) * (trial + 1))
                c._external_nodes.fill_(0.5 * trial)
                c._external_wrench.fill_(0.75 * trial)
                # Every real trial must overwrite old candidate contents.
                c._candidate.history.previous.fill_(-1000)
            run = helper.bind(current)
            run()
            eager(reference, baseline)
            for (name, array), (_, expected_array) in zip(
                _arrays(captured._candidate), _arrays(reference._candidate), strict=True
            ):
                np.testing.assert_array_equal(array.numpy(), expected_array.numpy(), err_msg=name)
            np.testing.assert_array_equal(current.body_f.numpy(), baseline.body_f.numpy())
            np.testing.assert_array_equal(captured._forces.numpy(), reference._forces.numpy())
        if self.device.is_cuda:
            self.assertEqual(helper.captures, 2)
            self.assertEqual(helper.launches, 11)
        else:
            self.assertEqual(helper.captures, 0)

    def test_rebinding_changed_namespace_storage_selects_a_new_graph(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA capture storage regression")
        c, incoming = coupler(self.device), state(self.device)
        helper = CapturedTrialPreparation(c)
        run = helper.bind(incoming)
        run()
        run()
        self.assertEqual(helper.captures, 1)
        old = incoming.history.previous
        incoming.history.previous = wp.full(3, 47.0, dtype=wp.float64, device=self.device)
        helper.bind(incoming)()
        np.testing.assert_array_equal(c._candidate.history.previous.numpy(), [47.0] * 3)
        self.assertEqual(helper.captures, 2)
        self.assertEqual(len(old), 3)

    def test_disabled_or_gradient_arrays_retain_eager_execution(self):
        c, incoming = coupler(self.device), state(self.device)
        helper = CapturedTrialPreparation(c)
        c.capture_fem = False
        for _ in range(2):
            helper.bind(incoming)()
        c.capture_fem = True
        incoming.history.previous.requires_grad = True
        for _ in range(2):
            helper.bind(incoming)()
        self.assertEqual(helper.captures, 0)
        self.assertEqual(helper.launches, 0)

    def test_missing_history_preserves_state_assign_error(self):
        c, incoming = coupler(self.device), state(self.device)
        c._candidate.history.previous = None
        with self.assertRaisesRegex(ValueError, "history.previous"):
            CapturedTrialPreparation(c).bind(incoming)()

    def test_graph_cache_is_bounded(self):
        if not self.device.is_cuda:
            self.skipTest("CUDA capture cache regression")
        c = coupler(self.device)
        helper = CapturedTrialPreparation(c)
        for _ in range(12):
            helper.bind(state(self.device))()
        self.assertEqual(helper.captures, 11)
        self.assertEqual(len(helper.graphs), 8)


if __name__ == "__main__":
    unittest.main()
