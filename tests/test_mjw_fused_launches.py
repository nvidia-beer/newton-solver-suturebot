"""Launches of the rigid step against those of ``mujoco_warp.step``.

Every kernel launch and memory operation of a step is a node with a fixed cost in the
CUDA graph that replays it. ``RigidStep.step`` runs each body-tree pass as one launch
instead of one per tree level, starts a constraint solve with eight launches and takes
five per Newton iteration instead of ten, and issues neither the memory operations of a
solver context per solve nor those of the sensor stage. Every other launch of the step
is MuJoCo Warp's own.

The counts are read from Warp's activity timing, which reports CUDA devices only. With
the solver loop as a loop in Python, a step issues exactly ``opt.iterations`` iterations,
so two steps of 0 and of 3 iterations split the counts into a fixed part and a part per
iteration. The counts of MuJoCo Warp's own step are asserted as well, so this test fails
when a MuJoCo Warp release changes its launches. That the fused launches leave the results
of MuJoCo Warp's is covered by ``test_warp_step.py``, ``test_warp_step_tree.py`` and
``test_warp_step_constraint_solver.py``.

    NEWTON_TEST_DEVICE=cuda:0 PYTHONPATH=tests python -m unittest test_mjw_fused_launches -v
"""

import unittest

import mujoco_warp as mjw
from _warp_step_support import CUDA, DeviceTestCase, activities, step_model

from newton_solver_suturebot._src.solvers.mujoco.warp_step import RigidStep

ITERATIONS = 3

# One launch per tree pass, and the fused launches of the constraint solve.
FUSED = {
    "subtree_com_fused": 1,
    "crb_fused": 1,
    "cfrc_backward_fused": 1,
    "solve_init_scale_efc": 1,
    "update_constraint_gradient_fused": 1 + ITERATIONS,
    "update_gradient_cholesky_fused": 1 + ITERATIONS,
}
# The launches of one constraint solve of the rigid step: eight at its start and five per iteration.
SOLVE = {
    "solve_init_dof": 1,
    "solve_init_scale_efc": 1,
    "solve_init_jaref_kernel": 1,
    "mul_m_kernel": 1,
    "update_constraint_gradient_fused": 1 + ITERATIONS,
    "update_gradient_JTDAJ_dense_tiled": 1 + ITERATIONS,
    "update_gradient_JTCJ_dense": 1 + ITERATIONS,
    "update_gradient_cholesky_fused": 1 + ITERATIONS,
    "linesearch_iterative_kernel": ITERATIONS,
}
# Kernels of MuJoCo Warp that the fused launches stand for.
TREE_LEVELS = ("subtree_com_acc", "crb_accumulate", "cfrc_backward")
REPLACED = (
    *TREE_LEVELS,
    "subtree_com_init",
    "subtree_div",
    "inertia_scale",
    "solve_init_efc",
    "update_constraint_efc",
    "update_constraint_init_qfrc_constraint_dense",
    "update_gradient_zero_grad_dot",
    "update_gradient_grad",
    "update_gradient_cholesky",
    "solve_done",
)


def _step_activities(name, rigid, iterations):
    """Activities of one step of a model, after the steps that load its kernels."""
    m, d = step_model(name)
    m.opt.graph_conditional = False
    m.opt.iterations = iterations
    m.opt.warn_overflow = 0
    step = RigidStep(m, d).step if rigid else lambda: mjw.step(m, d)
    for _ in range(2):
        step()
    return activities(step), len(m.body_tree)


@unittest.skipUnless(CUDA, "Warp reports the activities of CUDA devices only")
class RigidStepLaunchTests(DeviceTestCase):
    def test_step_issues_the_fused_launches_and_mujoco_warp_s_others(self):
        for name in ("gripper", "humanoid", "cables"):
            (stock, _), levels = _step_activities(name, False, ITERATIONS)
            (rigid, _), _ = _step_activities(name, True, ITERATIONS)
            with self.subTest(model=name):
                self.assertGreater(levels, 2)
                for kernel, count in FUSED.items():
                    self.assertEqual(rigid[kernel], count, kernel)
                    self.assertEqual(stock[kernel], 0, kernel)
                for kernel in REPLACED:
                    self.assertEqual(rigid[kernel], 0, kernel)
                    self.assertGreater(stock[kernel], 0, kernel)
                for kernel, count in SOLVE.items():
                    self.assertEqual(rigid[kernel], count, kernel)
                self.assertEqual(sum(SOLVE.values()), 8 + 5 * ITERATIONS)
                # MuJoCo Warp launches every tree level, the one of the world body included
                for kernel in TREE_LEVELS:
                    self.assertEqual(stock[kernel], levels, kernel)
                # the solve multiplies qacc by M once; the products of the search directions moved into
                # the launch that forms them
                self.assertEqual(stock["mul_m_kernel"], 1 + ITERATIONS)
                self.assertEqual(rigid["mul_m_kernel"], 1)
                changed = {*FUSED, *REPLACED, "mul_m_kernel"}
                others = {kernel: count for kernel, count in stock.items() if kernel not in changed}
                self.assertEqual({kernel: count for kernel, count in rigid.items() if kernel not in changed}, others)
                self.assertGreater(len(others), 20)

    def test_launches_per_step_and_per_solver_iteration(self):
        for name in ("gripper", "humanoid"):
            counts = {}
            for rigid in (False, True):
                for iterations in (0, ITERATIONS):
                    counts[rigid, iterations], levels = _step_activities(name, rigid, iterations)
            with self.subTest(model=name):
                kernels = {key: sum(value[0].values()) for key, value in counts.items()}
                # linesearch, constraint rows with qfrc_constraint and gradient, J'DJ, cone Hessian,
                # Cholesky with mv and termination
                self.assertEqual(kernels[True, ITERATIONS] - kernels[True, 0], 5 * ITERATIONS)
                self.assertEqual(kernels[False, ITERATIONS] - kernels[False, 0], 10 * ITERATIONS)
                # One launch stands for the level launches of com_pos and the two around them, one
                # each for those of crb and rne, and the start of a solve takes four launches less.
                saved = (levels + 2 - 1) + 2 * (levels - 1) + 4
                self.assertEqual(kernels[False, 0] - kernels[True, 0], saved)
                for iterations in (0, ITERATIONS):
                    stock, rigid = counts[False, iterations][1], counts[True, iterations][1]
                    # MuJoCo Warp fills the count of unconverged worlds of every solve
                    self.assertEqual(stock["built-in kernel"], 1)
                    self.assertEqual(rigid["built-in kernel"], 0)
                    # the two zeroed arrays of a solver context and the empty sensor data
                    self.assertEqual(stock["memset"] - rigid["memset"], 3)
                    # the copy of cinert that starts crb
                    self.assertEqual(stock["memcpy"] - rigid["memcpy"], 1)
                    self.assertEqual(rigid["graph"] + stock["graph"], 0)


if __name__ == "__main__":
    unittest.main()
