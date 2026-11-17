# SPDX-License-Identifier: Apache-2.0
"""Trial budget on the host-controlled trial loop of the grasp scene (native rigid backend).

The scene runs SubcycledMuJoCo with both tools, the shaft and surface preconditioners, the
preconditioned Anderson correction, tunnel multipliers and host wrench reuse. Its first
coupled step, with the needle pressed on the pad, evaluates every trial eagerly, as the
event oracle does: the surface event of a trial is in the trial fields of the interaction.

Two fixtures, each with its hard case stated here:

- The needle in the grasp of the reset (``_press``): the jaws hold the wire as the scene's
  reset seats them. The scene's loop converges on it in a few trials (3 on the CPU with the
  fitted capsule set, measured), so only the smallest budget ends it.
- The needle free (``_press(..., free=True)``): both jaws stand at the limit of their hinges
  and are commanded there, so nothing but the pad acts on the needle (the tests check that no
  tool touches it) and no number of this fixture depends on how the jaws rest on the wire. A
  free needle answers the tip force with its mass alone, which is the hard case of the trial
  loop. The scene's loop (surface metric, Anderson correction) needs 6 trials for it, with the
  first trials far above the tip force of the converged step (2.24 and 1.53 mN against
  1.29 mN). Without the surface metric the residual does not settle: after ten trials it
  alternates between 0.45 and 0.70 mN and the loop uses up the 160 trials of the coupler.
  Only the trial budget ends that loop. (All numbers measured on the CPU, 2026-11-17.)
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from test_grasp_scene import _scene

from newton_solver_suturebot._src.solvers.coupled.coupling import NeedleCouplingNotConverged
from newton_solver_suturebot.examples.grasp import _grasp_scene

_FIELDS = ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd")
_CUDA = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")).is_cuda


def _press(scene, puncture_force, *, free=False):
    """Press the needle tip on the pad; the tip force of the first step is about 2 mN.

    ``free``: the needle is not held. Both jaws are reset at the limit of their hinges and
    commanded there, so that nothing but the pad acts on the needle.
    """
    pose = scene.plan.pose(0).copy()
    pose[2] += 0.6 * scene.geometry.thickness
    if free:
        tool = scene.graspers[0]
        limit = 0.5 * tool.config.maximum_opening_rad
        with patch.object(_grasp_scene, "jaw_opening_on_needle", return_value=(limit, limit)):
            scene._initialize_pose(pose)
        scene.jaw_opening_target = tool.config.maximum_opening_rad
        tool.set_aperture(scene.control, 1.0)
    else:
        scene._initialize_pose(pose)
    scene.interaction.puncture_force = puncture_force
    scene.solver.refresh_tangent(scene.state_0.particle_q, scene.sim_dt)


def _tool_clearance(scene):
    """Smallest gap between the needle wire and the capsules of both tools in the reset state [m].

    Negative: a capsule lies on the wire by that much.
    """
    from newton_solver_suturebot.examples.grasp._dvrk_grasper import _segment_distance

    poses = scene.state_0.body_q.numpy()
    needle = wp.transform(*poses[scene.needle_body])
    line = np.asarray([np.asarray(wp.transform_point(needle, wp.vec3(*point))) for point in scene.geometry.centerline])
    segments = np.stack((line[:-1], line[1:]), axis=1)
    clearance = np.inf
    for tool in scene.graspers:
        for row, body in enumerate(tool.capsule_bodies):
            transform = wp.transform(*poses[body])
            ends = np.asarray(
                [np.asarray(wp.transform_point(transform, wp.vec3(*end))) for end in tool.capsule_endpoints[row]]
            )
            gap = min(_segment_distance(ends, segment) for segment in segments)
            clearance = min(clearance, gap - tool.capsule_radii[row] - scene.geometry.thickness)
    return float(clearance)


def _record(scene, hook):
    """Record every trial with a primal evaluation: residual of the trial before, its load, its tip force.

    ``"restore"`` records through ``_restore_native``, which the loop calls once per such trial
    and which does not select a trial implementation: the loop is the one the scene runs. The
    tip force it finds there is that of the trial before, like the residual.
    ``"evaluate"`` records after ``_evaluate``, where the tip force of the trial is known. A
    coupler with its own ``_evaluate`` gets no surface metric, so that loop runs with the shaft
    preconditioner alone, as the scene does whenever the surface metric is unavailable.
    """
    coupler, trials = scene.coupler, []
    if hook == "restore":
        restore = coupler._restore_native

        def restored():
            trials.append((coupler.residual, coupler._trial_load.numpy().copy(), scene.interaction.last_tip_force))
            return restore()

        coupler._restore_native = restored
    else:
        evaluate = coupler._evaluate

        def evaluated(*args, **kwargs):
            before = coupler.residual
            evaluate(*args, **kwargs)
            trials.append((before, coupler._trial_load.numpy().copy(), scene.interaction.last_tip_force))

        coupler._evaluate = evaluated
    return trials


def _run(budget, *, hook, free=False, steps=1, relative_tolerance=None, puncture_force=1.0):
    """One pressed scene: what its first coupled step did, and the trials of the steps after it.

    A loop without a budget that uses up the trials of the coupler is recorded as it ended
    (``converged`` false): the step is rejected and nothing is accepted. With a budget the
    rejection is an error of the test.
    """
    with _scene() as (scene, _):
        _press(scene, puncture_force, free=free)
        clearance = _tool_clearance(scene)
        coupler, interaction = scene.coupler, scene.interaction
        coupler.trial_budget = budget
        if relative_tolerance is not None:
            coupler.relative_tolerance = relative_tolerance
        trials = _record(scene, hook)
        converged = True
        with patch.object(interaction, "accept", wraps=interaction.accept) as accept:
            try:
                coupler.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
            except NeedleCouplingNotConverged:
                if budget is not None:
                    raise
                converged = False
            accepts = accept.call_count
        transfer = getattr(scene.rigid_solver, "_native_transfer", None)
        tip_forces = [force for _, _, force in trials]
        if hook == "restore":
            tip_forces = [*tip_forces[1:], interaction.last_tip_force]
        result = {
            "converged": converged,
            "tool_clearance": clearance,
            "wire_radius": scene.geometry.thickness,
            "trial_limit": coupler.max_iterations,
            "trials": coupler.iterations,
            "residuals": [residual for residual, _, _ in trials[1:]] + [coupler.residual],
            "loads": [load for _, load, _ in trials],
            "tip_forces": tip_forces,
            "accepts": accepts,
            "keeps": coupler.coupling_budget_keeps,
            "keep_residual": coupler.coupling_budget_residual,
            "keep_events": dict(coupler.coupling_budget_events),
            "residual": coupler.residual,
            "load": coupler._load.numpy().copy(),
            "value": coupler._value.numpy().copy(),
            "trial_load": coupler._trial_load.numpy().copy(),
            "state": {name: getattr(scene.state_1, name).numpy().copy() for name in _FIELDS},
            "candidate": {name: getattr(coupler._candidate, name).numpy().copy() for name in _FIELDS},
            "alm_open": interaction.tunnel_alm._open,
            "dual_updates": interaction.tunnel_alm.total_dual_updates,
            "force_balance": interaction.force_balance_error,
            "torque_balance": interaction.torque_balance_error,
            "tolerances": (coupler.absolute_tolerance, coupler.relative_tolerance),
            "repeated_active": None if transfer is None else transfer._repeated_active,
            "transaction": coupler._host_wrench_transaction,
            "launches": coupler.coupling_graph_launches,
            "status": interaction.status,
            "punctured": interaction.punctured,
            "punctures": interaction.puncture_count,
            "events": [event["kind"] for event in interaction.events],
            "following": [],
        }
        for _ in range(steps - 1):
            scene.state_0, scene.state_1 = scene.state_1, scene.state_0
            coupler.step(scene.state_0, scene.state_1, scene.control, scene.sim_dt)
            result["following"].append(coupler.iterations)
        return result


def _best(residuals):
    """Index of the trial the rule names: smallest residual; of equal ones the later."""
    return max(range(len(residuals)), key=lambda index: (-residuals[index], index))


class _HostLoop:
    hook = None
    free = False
    # Whether the loop without a budget converges within the trials of the coupler.
    converges = True

    @classmethod
    def setUpClass(cls):
        cls.reference = _run(None, hook=cls.hook, free=cls.free)

    def test_reference_loop_runs_on_the_host_and_keeps_nothing(self):
        reference = self.reference
        # The hard case of the fixture, as the module text states it.
        self.assertEqual(reference["converged"], self.converges)
        if self.free:
            # No capsule of a tool is nearer the wire than two wire diameters (2.02 mm clear, measured).
            self.assertGreater(reference["tool_clearance"], 4.0 * reference["wire_radius"])
        else:
            # The bite: capsules of the jaws lie on the wire.
            self.assertLessEqual(reference["tool_clearance"], 0.0)
        self.assertEqual(reference["launches"], 0)
        self.assertGreater(reference["trials"], 2)
        self.assertEqual((reference["keeps"], reference["keep_residual"], reference["keep_events"]), (0, 0.0, {}))
        self.assertEqual(reference["dual_updates"], 0)

    def test_budget_keeps_the_trial_the_rule_names(self):
        reference = self.reference
        branches = set()
        budgets = [budget for budget in (2, 3, 4, 5, 6, 12) if budget < reference["trials"]]
        self.assertTrue(budgets)
        for budget in budgets:
            with self.subTest(budget=budget):
                result = _run(budget, hook=self.hook, free=self.free, steps=3)
                self.assertEqual(result["launches"], 0)
                # One multiplier epoch: the best trial is the best of all trials up to the budget.
                self.assertEqual(result["dual_updates"], 0)
                self.assertEqual(len(result["loads"]), result["trials"])
                if not _CUDA:
                    # Up to the budget the loop is the loop without one, bit for bit.
                    for number in range(budget):
                        np.testing.assert_array_equal(
                            result["loads"][number], reference["loads"][number], err_msg=str(number)
                        )
                    self.assertEqual(result["residuals"][:budget], reference["residuals"][:budget])
                best = _best(result["residuals"][:budget])
                if best == budget - 1:
                    branches.add("last")
                    self.assertEqual(result["trials"], budget)
                    np.testing.assert_array_equal(result["trial_load"], result["loads"][budget - 1])
                else:
                    branches.add("applied once more")
                    self.assertEqual(result["trials"], budget + 1)
                    # One more trial at the best load, bit for bit, and nothing after it.
                    np.testing.assert_array_equal(result["loads"][budget], result["loads"][best])
                    np.testing.assert_array_equal(result["trial_load"], result["loads"][best])
                    self.assertFalse(np.array_equal(result["loads"][budget - 1], result["loads"][best]))
                # The acceptance of a converged trial, once.
                self.assertEqual((result["accepts"], result["keeps"]), (1, 1))
                self.assertEqual(result["keep_residual"], result["residual"])
                for name in _FIELDS:
                    np.testing.assert_array_equal(result["state"][name], result["candidate"][name], err_msg=name)
                # The carried load is the load the kept state was computed with.
                np.testing.assert_array_equal(result["load"], result["trial_load"])
                self.assertEqual(float(np.abs(result["value"] - result["trial_load"]).max()), result["residual"])
                self.assertFalse(result["alm_open"])
                self.assertFalse(result["repeated_active"])
                self.assertIsNone(result["transaction"])
                # Above the tolerance: a kept trial, not a converged one.
                absolute, relative = result["tolerances"]
                scale = max(np.abs(result["value"]).max(), np.abs(result["trial_load"]).max())
                self.assertGreater(result["residual"], absolute + relative * scale)
                # Needle and tissue were advanced with one load: the balance of a converged step
                # (2.3e-10 N and 2.1e-12 N m at a converged reference, measured).
                self.assertLess(result["force_balance"], 2e-8)
                self.assertLess(result["torque_balance"], 1e-9)
                # No surface event at this threshold, and the steps after a kept one run.
                self.assertEqual((result["keep_events"], result["status"]), ({}, "contact"))
                self.assertEqual(len(result["following"]), 2)
        if not _CUDA and self.hook == "evaluate":
            # Measured on the CPU: budgets 2, 3 and 5 keep their last trial, 4, 6 and 12 an earlier one.
            self.assertEqual(branches, {"last", "applied once more"})

    def test_example_defaults_end_the_loop_at_the_budget_or_one_trial_after_it(self):
        # Tolerance 1e-2 and budget 12: the defaults of the grasp example, also on this backend.
        result = _run(12, hook=self.hook, free=self.free, steps=3, relative_tolerance=1e-2)
        self.assertLessEqual(result["trials"], 13)
        self.assertEqual(result["accepts"], 1)
        self.assertLessEqual(result["keeps"], 1)


class SceneLoopTests(_HostLoop, unittest.TestCase):
    """The loop the scene runs: surface metric, Anderson correction, host wrench reuse."""

    hook = "restore"


class SceneLoopFreeNeedleTests(_HostLoop, unittest.TestCase):
    """The loop the scene runs, on the free needle: more trials, so more budgets end it."""

    hook = "restore"
    free = True

    def test_reference_loop_needs_more_trials_than_the_small_budgets(self):
        # 6 trials on the CPU (measured): the budgets 2 to 5 end the loop.
        self.assertGreater(self.reference["trials"], 4)


class LoopWithoutSurfaceMetricTests(_HostLoop, unittest.TestCase):
    """The free needle when the surface metric is unavailable: the residual does not settle."""

    hook = "evaluate"
    free = True
    converges = False

    def test_reference_loop_runs_past_the_default_budget(self):
        # The loop without a budget uses up the 160 trials of the coupler (measured on the CPU:
        # the residual alternates between 0.45 and 0.70 mN from the tenth trial on).
        self.assertGreater(self.reference["trials"], 13)
        self.assertEqual(self.reference["trials"], self.reference["trial_limit"])


class KeptSurfaceEventTests(unittest.TestCase):
    """A trial the host loop keeps is accepted with the surface event of that trial."""

    def test_kept_trial_commits_a_puncture_that_the_converged_step_does_not_decide(self):
        # The scene's loop on the free needle: it converges, and its first trials are far
        # above the tip force of the converged step.
        fixture = {"hook": "restore", "free": True}
        reference = _run(None, **fixture)
        self.assertTrue(reference["converged"])
        self.assertGreater(reference["tool_clearance"], 4.0 * reference["wire_radius"])
        self.assertEqual((reference["punctures"], reference["events"], reference["status"]), (0, [], "contact"))
        converged = reference["tip_forces"][-1]
        # The first trials overshoot the tip force of the converged step (2.24 and 1.53 mN
        # against 1.29 mN on the CPU, measured). The budget that keeps the trial with the
        # largest overshoot, and a puncture threshold halfway between that trial and the
        # converged step.
        overshoot = {
            budget: reference["tip_forces"][_best(reference["residuals"][:budget])]
            for budget in range(2, 7)
            if budget < reference["trials"]
        }
        budget = max(overshoot, key=overshoot.get)
        kept_force = overshoot[budget]
        self.assertGreater(kept_force, converged * (1 + 2e-3))
        threshold = 0.5 * (kept_force + converged)

        # Without a budget the step converges below the threshold: no puncture in this step.
        unbudgeted = _run(None, puncture_force=threshold, **fixture)
        self.assertTrue(unbudgeted["converged"])
        self.assertEqual(unbudgeted["trials"], reference["trials"])
        self.assertLess(unbudgeted["tip_forces"][-1], threshold)
        self.assertEqual((unbudgeted["punctures"], unbudgeted["events"], unbudgeted["status"]), (0, [], "contact"))
        self.assertEqual((unbudgeted["keeps"], unbudgeted["keep_events"]), (0, {}))

        # With the budget the kept trial is above it, and its puncture is committed with the step.
        kept = _run(budget, puncture_force=threshold, **fixture)
        self.assertLessEqual(kept["trials"], budget + 1)
        self.assertEqual((kept["accepts"], kept["keeps"]), (1, 1))
        self.assertGreaterEqual(kept["tip_forces"][-1], threshold)
        self.assertTrue(kept["punctured"])
        self.assertEqual((kept["punctures"], kept["events"], kept["status"]), (1, ["puncture"], "punctured"))
        # The coupler reports the event of the kept step, with the residual it was decided at.
        self.assertEqual(kept["keep_events"], {"puncture": 1})
        self.assertEqual(kept["keep_residual"], kept["residual"])
        absolute, relative = kept["tolerances"]
        scale = max(np.abs(kept["value"]).max(), np.abs(kept["trial_load"]).max())
        self.assertGreater(kept["residual"], absolute + relative * scale)


if __name__ == "__main__":
    unittest.main()
