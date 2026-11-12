"""The articulated shaft accelerator reads a native contact metric without changing physics."""

import os
import unittest

import mujoco
import numpy as np
import warp as wp

from newton_solver_suturebot._src.shaft_preconditioner import (
    _native_needle_mass,
    _rigid_blocks_from_mass,
    rigid_mass_action,
)


def native(*, sparse=False, contact=False):
    plane = '<geom type="plane" size="1 1 .01"/>' if contact else ""
    model = mujoco.MjModel.from_xml_string(f'''
    <mujoco>
      <option gravity="0 0 0" cone="elliptic" jacobian="{"sparse" if sparse else "dense"}" tolerance="1e-12"/>
      <worldbody>
        {plane}
        <body name="needle" pos="0 0 .099">
          <freejoint/><geom type="sphere" size=".1" mass=".2" condim="6" friction=".9 .03 .01"/>
        </body>
        <body name="tool" pos=".3 0 .2">
          <joint type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="capsule" fromto="0 0 0 0 0 .1" size=".01" mass=".1"/>
        </body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def integration_state(model, data):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, state, kind)
    return state


class NativeNeedleMassTests(unittest.TestCase):
    def test_free_needle_matches_world_mass_inertia_with_additional_articulated_tool(self):
        model, data = native()
        model.dof_damping[:6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        theta = 0.7
        data.qpos[3:7] = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        mujoco.mj_forward(model, data)
        h, length = 0.004, 0.02
        result = _native_needle_mass(model, data, 1, length, h)
        rotation = data.xmat[1].reshape(3, 3)
        expected = np.zeros((6, 6))
        expected[:3, :3] = np.diag(model.body_mass[1] + h * model.dof_damping[:3])
        expected[3:, 3:] = (
            rotation @ np.diag(model.body_inertia[1] + h * model.dof_damping[3:6]) @ rotation.T / length**2
        )
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)

    def test_active_native_contact_reduces_needle_compliance_in_dense_and_sparse_modes(self):
        results = []
        for sparse in (False, True):
            model, data = native(sparse=sparse, contact=True)
            self.assertGreater(data.nefc, 0)
            result = _native_needle_mass(model, data, 1, 0.02, 0.004)
            self.assertGreater(np.linalg.eigvalsh(result).min(), 0)
            self.assertGreater(result[2, 2], model.body_mass[1] * 2)
            np.testing.assert_allclose(result, result.T, rtol=0, atol=1e-12)
            results.append(result)
        np.testing.assert_allclose(results[0], results[1], rtol=1e-12, atol=1e-12)

    def test_sliding_elliptic_cone_hessian_preserves_spd_and_native_state(self):
        model, data = native(contact=True)
        data.qvel[0] = 1
        mujoco.mj_forward(model, data)
        self.assertIn(int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE), data.efc_state)
        before = integration_state(model, data)
        fields = {name: getattr(data, name).copy() for name in ("qacc", "efc_J", "efc_D", "efc_force", "efc_state")}
        old_hessians = [c.H.copy() for c in data.contact]
        scratch = mujoco.MjData(model)
        for _ in range(3):
            result = _native_needle_mass(model, data, 1, 0.02, 0.004, scratch)
            self.assertGreater(np.linalg.eigvalsh(result).min(), 0)
        np.testing.assert_array_equal(integration_state(model, data), before)
        for name, value in fields.items():
            np.testing.assert_array_equal(getattr(data, name), value)
        for old, contact in zip(old_hessians, data.contact, strict=True):
            np.testing.assert_array_equal(contact.H, old)

    def test_gpu_operator_representation_retains_translation_rotation_cross_terms(self):
        model, data = native(contact=True)
        matrix = _native_needle_mass(model, data, 1, 0.02, 0.004)
        self.assertGreater(np.linalg.norm(matrix[:3, 3:]), 1e-6)
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        blocks = wp.array(
            np.asarray([matrix[:3, :3], matrix[3:, 3:], matrix[:3, 3:], matrix[3:, :3]], np.float32),
            dtype=wp.mat33,
            device=device,
        )
        vector = np.random.default_rng(7).normal(size=(2, 3)).astype(np.float32)
        value = wp.array(vector, dtype=wp.vec3, device=device)
        result = wp.zeros(2, dtype=wp.vec3, device=device)
        wp.launch(rigid_mass_action, 2, [blocks, value, result], device=device)
        np.testing.assert_allclose(result.numpy().ravel(), matrix @ vector.ravel(), rtol=2e-6, atol=2e-6)

    def test_condensed_metric_matches_native_acceleration_force_derivative(self):
        for sliding in (False, True):
            with self.subTest(sliding=sliding):
                model, data = native(contact=True)
                data.qvel[0] = 1.0 if sliding else 0.0
                mujoco.mj_forward(model, data)
                length, epsilon = 0.02, 1e-6
                # Near-zero h isolates native contact inertia from the added
                # velocity damping approximation in the search direction.
                matrix = _native_needle_mass(model, data, 1, length, 1e-12)
                jacobian = np.zeros((6, model.nv))
                mujoco.mj_jacBodyCom(model, data, jacobian[:3], jacobian[3:], 1)
                jacobian[3:] *= length
                columns = []
                for axis in range(6):
                    plus, minus = mujoco.MjData(model), mujoco.MjData(model)
                    mujoco.mj_copyData(plus, model, data)
                    mujoco.mj_copyData(minus, model, data)
                    force = epsilon * (1 if axis < 3 else length)
                    plus.xfrc_applied[1, axis] += force
                    minus.xfrc_applied[1, axis] -= force
                    mujoco.mj_forward(model, plus)
                    mujoco.mj_forward(model, minus)
                    columns.append(jacobian @ (plus.qacc - minus.qacc) / (2 * epsilon))
                expected = np.asarray(columns).T
                actual = np.linalg.inv(matrix)
                self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-5)

    def test_metric_that_loses_positive_definiteness_in_float32_is_rejected(self):
        matrix = np.eye(6)
        matrix[0, 1] = matrix[1, 0] = 1.0
        matrix[1, 1] = 1.0 + 1e-8
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)
        with self.assertRaises(np.linalg.LinAlgError):
            _rigid_blocks_from_mass(matrix)
        matrix[0, 0] = 1e300
        with self.assertRaises(ValueError):
            _rigid_blocks_from_mass(matrix)

    def test_invalid_tool_damping_outside_the_needle_is_rejected(self):
        model, data = native()
        self.assertGreater(model.nv, 6)
        for value in (-0.1, np.nan):
            model.dof_damping[-1] = value
            with self.subTest(damping=value), self.assertRaises(ValueError):
                _native_needle_mass(model, data, 1, 0.02, 0.004)

    def test_invalid_numerical_scales_are_rejected(self):
        model, data = native()
        for length, h in ((0, 0.1), (0.02, 0), (np.nan, 0.1), (0.02, np.inf)):
            with self.subTest(length=length, dt=h), self.assertRaises(ValueError):
                _native_needle_mass(model, data, 1, length, h)


if __name__ == "__main__":
    unittest.main()
