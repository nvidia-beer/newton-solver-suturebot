import os
import unittest

import numpy as np
import warp as wp

from newton_solver_suturebot._src.solvers.mujoco.rigid_contact_metric import (
    CapturedRigidContactMetric,
    _prepare_factor,
    condense,
)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        wp.set_device(cls.device)

    def run_matrix(self, h, b, damping=None, dt=0.004):
        n = h.shape[0]
        damping = np.zeros((1, n), np.float32) if damping is None else damping
        h_input = wp.array(np.asarray([h], np.float32), device=self.device)
        damp_input = wp.array(damping, device=self.device)
        jac = wp.array(np.asarray(b, np.float64), device=self.device)
        factor = wp.zeros((n, n), dtype=wp.float64, device=self.device)
        response = wp.zeros((6, 6), dtype=wp.float64, device=self.device)
        small = wp.zeros_like(response)
        inverse = wp.zeros_like(response)
        blocks = wp.zeros(4, dtype=wp.mat33, device=self.device)
        valid = wp.zeros(1, dtype=int, device=self.device)
        # The kernel reads the interval from a one-element device array (StepTime.seconds).
        interval = wp.array([dt], dtype=float, device=self.device)
        wp.launch(_prepare_factor, (n, n), [h_input, damp_input, interval, factor], device=self.device)
        condense(jac, factor, response, small, inverse, blocks, valid)
        result = blocks.numpy()
        return int(valid.numpy()[0]), np.block([[result[0], result[2]], [result[3], result[1]]])

    def test_spd_condensation_matches_numpy_and_ignores_lower_garbage(self):
        rng = np.random.default_rng(7)
        for n in (6, 18, 26):
            with self.subTest(n=n):
                a = rng.normal(size=(n, n))
                h = np.asarray(a @ a.T + np.eye(n), np.float32)
                b = rng.normal(size=(6, n))
                damping = rng.uniform(0, 10, (1, n)).astype(np.float32)
                expected = np.linalg.inv(b @ np.linalg.solve(h + 0.004 * np.diag(damping[0]), b.T))
                garbage = h.copy()
                garbage[np.tril_indices(n, -1)] = np.nan
                valid, actual = self.run_matrix(garbage, b, damping)
                self.assertEqual(valid, 1)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
                np.linalg.cholesky(actual.astype(float))

    def test_bad_metric_is_rejected(self):
        for h, b in ((-np.eye(6), np.eye(6)), (np.eye(6), np.zeros((6, 6))), (np.eye(6) * np.nan, np.eye(6))):
            self.assertEqual(self.run_matrix(h, b)[0], 0)

    def test_real_body_jacobian_and_capture_replay(self):
        import mujoco
        import mujoco_warp as mjw

        from newton_solver_suturebot._src.solvers.mujoco.warp_step import RigidStep

        native = mujoco.MjModel.from_xml_string("""<mujoco>
          <option integrator="implicitfast" solver="Newton" cone="elliptic" timestep="0.001"/>
          <worldbody><geom type="plane" size="1 1 .1"/>
            <body pos="0 0 .09"><freejoint/>
              <geom type="sphere" size=".1" mass=".3"/>
            </body>
          </worldbody></mujoco>""")
        data = mujoco.MjData(native)
        mujoco.mj_forward(native, data)
        m = mjw.put_model(native)
        d = mjw.put_data(native, data, nworld=1)
        blocks = wp.zeros(4, dtype=wp.mat33, device=self.device)
        rigid_step = RigidStep(m, d)
        rigid_step.step()
        metric = CapturedRigidContactMetric(rigid_step, 0.02, blocks)
        self.assertIs(metric.context, rigid_step.constraint_solve.context)
        metric.prepare(0.004)
        self.assertEqual(int(metric.valid.numpy()[0]), 1)
        h = metric.context.h.numpy()[0, : native.nv, : native.nv].astype(float)
        h = np.triu(h) + np.triu(h, 1).T
        h += 0.004 * np.diag(m.dof_damping.numpy()[0])
        b = metric.jacobian.numpy()
        expected = np.linalg.inv(b @ np.linalg.solve(h, b.T))
        result = blocks.numpy()
        actual = np.block([[result[0], result[2]], [result[3], result[1]]])
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-6)
        # The GPU kinematics are from the beginning of this microstep. Native
        # data still represents precisely that pose; comparing avoids an extra
        # CPU forward or a misleading end-of-step frame comparison.
        jacp, jacr = np.zeros((3, native.nv)), np.zeros((3, native.nv))
        mujoco.mj_jacBodyCom(native, data, jacp, jacr, 1)
        np.testing.assert_allclose(b, np.vstack([jacp, 0.02 * jacr]), rtol=1e-6, atol=1e-8)
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                metric.prepare(0.004)
            for _ in range(3):
                blocks.zero_()
                wp.capture_launch(capture.graph)
                self.assertEqual(int(metric.valid.numpy()[0]), 1)
                np.testing.assert_array_equal(blocks.numpy(), result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
