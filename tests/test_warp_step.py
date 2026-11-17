"""``RigidStep`` against ``mujoco_warp.step``.

``RigidStep.step`` and ``mujoco_warp.step`` get one Model and two equal Data and must
leave the same Data. On the CPU both run the same operations in the same order, so
every array of Data is compared bit for bit after each of 50 steps of two rollouts.

On CUDA MuJoCo Warp's step does not repeat itself bit for bit: contacts and constraint
rows take their slots, and sums their order, from races between threads, and contact
dynamics amplify the differences from step to step. Every step there starts from
equal Data, and the integration state is compared to 1e-3 of the largest value, the
tolerance of the other MuJoCo Warp comparisons in these tests.

    PYTHONPATH=tests python -m unittest test_warp_step -v
    NEWTON_TEST_DEVICE=cuda:0 PYTHONPATH=tests python -m unittest test_warp_step -v
"""

import unittest

import mujoco_warp as mjw
import numpy as np
import warp as wp
from _warp_step_support import (
    CUDA,
    DEVICE,
    DeviceTestCase,
    allocation_tracker,
    allocations,
    clone_data,
    count_down,
    data_arrays,
    step_model,
)
from mujoco_warp._src.types import BroadphaseType, ConstraintType, DeterminismType, DisableBit, EnableBit

from newton_solver_suturebot._src.solvers.mujoco.warp_step import RigidStep, companion_data
from newton_solver_suturebot._src.solvers.mujoco.warp_step.constraint_solver import NewtonSolve
from newton_solver_suturebot._src.solvers.mujoco.warp_step.scratch import StepScratch
from newton_solver_suturebot._src.solvers.mujoco.warp_step.tree import TreePasses

STEPS = 50

# Integration state that a step advances, and the acceleration it solves for.
STATE = ("qpos", "qvel", "act", "time", "qacc", "qacc_warmstart")
# What a caller sets between steps.
INPUTS = ("ctrl", "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat")


def _energy(m):
    """Energies are computed in the position and velocity stages."""
    m.opt.enableflags |= EnableBit.ENERGY


def _rne_postconstraint(m):
    """Body accelerations and interaction forces after the solve, which the rigid solver asks for."""
    m.opt.run_rne_postconstraint = True


def _control_callback(m):
    """A control law that replaces the actuator targets before the actuation stage."""
    m.callback.control = lambda m, d: d.ctrl.fill_(0.25)


def _control_callback_without_actuation(m):
    """Actuation is disabled, and the control law with it."""
    _control_callback(m)
    m.opt.disableflags |= DisableBit.ACTUATION


# name, worlds, edit of the device model
CASES = (
    ("arm", 1, None),
    ("gripper", 1, None),
    ("boxes", 1, None),
    ("humanoid", 1, None),
    ("cables", 1, None),
    ("gripper", 2, None),
    ("arm", 1, _energy),
    ("boxes", 1, _rne_postconstraint),
    ("gripper", 1, _control_callback),
    ("gripper", 1, _control_callback_without_actuation),
)


def _kick(d, rng):
    """Change the velocities, differently in every world, so that contacts slide and a solve needs iterations."""
    qvel = d.qvel.numpy()
    d.qvel.assign(qvel + rng.uniform(-0.2, 0.2, qvel.shape).astype(np.float32))


def _state(d):
    return {name: getattr(d, name).numpy().copy() for name in STATE if getattr(d, name).size}


def _census(d, counts):
    """Largest number of contacts, constraint rows, limit rows and solver iterations over the steps."""
    active = np.arange(d.efc.type.shape[1])[None, :] < d.nefc.numpy()[:, None]
    kind = d.efc.type.numpy()
    found = {
        "contacts": int(d.nacon.numpy()[0]),
        "rows": int(d.nefc.numpy().max()),
        "joint limits": int((active & (kind == ConstraintType.LIMIT_JOINT)).sum(axis=1).max()),
        "tendon limits": int((active & (kind == ConstraintType.LIMIT_TENDON)).sum(axis=1).max()),
        "iterations": int(d.solver_niter.numpy().max()),
    }
    for name, value in found.items():
        counts[name] = max(counts.get(name, 0), value)


class RigidStepTests(DeviceTestCase):
    def _assert_close(self, actual, expected, **context):
        """Integration state within the CUDA tolerance; the clock exactly."""
        for name, value in expected.items():
            with self.subTest(field=name, **context):
                self.assertTrue(np.isfinite(value).all())
                bound = 0.0 if name == "time" else 1e-3 * float(np.abs(value).max())
                np.testing.assert_allclose(actual[name], value, rtol=0.0, atol=bound, err_msg=name)

    def _assert_exercised(self, counts, name, m, d):
        with self.subTest(model=name):
            self.assertLessEqual(counts["rows"], d.njmax)
            self.assertGreater(counts["iterations"], 1)
            if name == "cables":
                self.assertEqual((m.na, m.ntendon, m.ncam, m.nlight), (1, 2, 1, 1))
                self.assertGreater(counts["tendon limits"], 0)
            else:
                self.assertGreater(counts["contacts"], 0)
                self.assertGreater(counts["joint limits"], 0)

    def test_steps_match_mujoco_warp(self):
        for name, worlds, edit in CASES:
            m, d = step_model(name, worlds, edit)
            reference = clone_data(d)
            rigid = RigidStep(m, d)
            rng, counts = np.random.default_rng(3), {}
            for step in range(STEPS):
                if step % 10 == 0:
                    _kick(d, rng)
                    wp.copy(reference.qvel, d.qvel)
                if CUDA:
                    # two CUDA rollouts drift apart; every step starts from equal Data
                    reference = clone_data(d)
                rigid.step()
                mjw.step(m, reference)
                _census(reference, counts)
                context = {"model": name, "worlds": worlds, "edit": edit and edit.__name__, "step": step}
                if CUDA:
                    self._assert_close(_state(d), _state(reference), **context)
                    continue
                different = [
                    field
                    for (field, actual), (_, expected) in zip(data_arrays(d), data_arrays(reference), strict=True)
                    if actual.numpy().tobytes() != expected.numpy().tobytes()
                ]
                with self.subTest(**context):
                    self.assertEqual(different, [])
                    self.assertTrue(all(np.isfinite(value).all() for value in _state(d).values()))
            with self.subTest(model=name, worlds=worlds, edit=edit and edit.__name__):
                self.assertGreater(len(list(data_arrays(d))), 100)
                energy = d.energy.numpy()
                if edit is _energy:
                    self.assertTrue((energy[:, 1] > 0.0).all())
                elif not CUDA:
                    np.testing.assert_array_equal(energy, 0.0)
                if edit is _rne_postconstraint:
                    self.assertGreater(float(np.abs(d.cfrc_ext.numpy()).max()), 0.0)
                if edit is _control_callback:
                    np.testing.assert_array_equal(d.ctrl.numpy(), 0.25)
                if edit is _control_callback_without_actuation:
                    self.assertFalse((d.ctrl.numpy() == 0.25).any())
            self._assert_exercised(counts, name, m, d)

    def test_companion_data_shares_what_steps_carry_and_nothing_else(self):
        m, d = step_model("gripper")
        companion = companion_data(d)
        carried = {f"d.{name}" for name in INPUTS + ("time", "qpos", "qvel", "act", "qacc_warmstart", "overflow")}
        shared = {
            name for (name, own), (_, other) in zip(data_arrays(companion), data_arrays(d), strict=True) if own is other
        }
        self.assertTrue(carried <= shared, sorted(carried - shared))
        # the rest of what is shared is the integration state that this model has no entries of
        self.assertEqual({name for name in shared - carried if getattr(d, name[2:]).size}, set())
        for (name, own), (_, other) in zip(data_arrays(companion), data_arrays(d), strict=True):
            if own is not other and own.size:
                with self.subTest(array=name):
                    self.assertNotEqual(own.ptr, other.ptr)
                    self.assertEqual((own.shape, own.dtype), (other.shape, other.dtype))
                    self.assertEqual(own.numpy().tobytes(), other.numpy().tobytes())

    def test_repeated_first_step_keeps_its_state_stages(self):
        """Intervals of three steps that restart from one state with another wrench, as coupled trials do.

        The first step runs on a companion Data. Its state stages run once; every repetition
        issues ``step(state_stages=False)`` there and two whole steps on the main Data. A plain
        ``RigidStep`` on a third Data repeats every stage and must end in the same state.
        """
        for name, worlds in (("gripper", 1), ("boxes", 1), ("humanoid", 1), ("gripper", 2)):
            m, d = step_model(name, worlds)
            reference = clone_data(d)
            main, first, plain = RigidStep(m, d), RigidStep(m, companion_data(d)), RigidStep(m, reference)
            rng = np.random.default_rng(5)
            for warm in range(12):
                if warm % 6 == 0:
                    _kick(d, rng)
                    wp.copy(reference.qvel, d.qvel)
                # an eager interval of the split form warms both steps and sizes their pools
                first.step()
                main.step()
                for _ in range(2):
                    plain.step()
                if CUDA:
                    for field in STATE:
                        wp.copy(getattr(reference, field), getattr(d, field))
            start = _state(d)
            body = min(m.nbody - 1, 2)

            def restore(data, trial, start=start, body=body):
                for field, value in start.items():
                    getattr(data, field).assign(value)
                wrench = np.zeros(data.xfrc_applied.shape + (6,), dtype=np.float32)
                wrench[:, body] = 0.4 * (trial + 1) * np.array([1.0, -0.5, 0.25, 0.01, 0.02, -0.01])
                data.xfrc_applied.assign(wrench)
                data.overflow.zero_()

            def interval(first=first, main=main):
                first.step(state_stages=False)
                main.step()
                main.step()

            ends = []
            for trial in range(4):
                restore(d, trial)
                restore(reference, trial)
                if trial == 0:
                    first.state_stages()
                if CUDA and trial == 1:
                    # the repetitions replay from a graph, as the trials of a coupled interval do
                    with wp.ScopedCapture() as capture:
                        interval()
                if CUDA and trial >= 1:
                    wp.capture_launch(capture.graph)
                else:
                    interval()
                for _ in range(3):
                    plain.step()
                context = {"model": name, "worlds": worlds, "trial": trial}
                if CUDA:
                    self._assert_close(_state(d), _state(reference), **context)
                else:
                    for field, value in _state(reference).items():
                        with self.subTest(field=field, **context):
                            self.assertEqual(_state(d)[field].tobytes(), value.tobytes())
                ends.append(_state(d)["qvel"])
            with self.subTest(model=name, worlds=worlds):
                # the wrench of a trial reaches its result
                self.assertGreater(float(np.abs(ends[1] - ends[0]).max()), 0.0)
                np.testing.assert_array_equal(d.time.numpy(), reference.time.numpy())

    def test_step_without_state_stages_needs_them_once(self):
        m, d = step_model("arm")
        rigid = RigidStep(m, d)
        rigid.step()
        signature = rigid.signature
        rigid.state_stages()
        rigid.step(state_stages=False)
        rigid.step()
        # either form asks each pool for the requests it was sized with
        self.assertEqual(rigid.signature, signature)

    def test_step_owns_its_parts_and_calls_the_hook_once(self):
        m, d = step_model("gripper")
        rigid = RigidStep(m, d)
        self.assertIsInstance(rigid.tree, TreePasses)
        self.assertIsInstance(rigid.constraint_solve, NewtonSolve)
        self.assertIsInstance(rigid.scratch, StepScratch)
        self.assertIs(rigid.constraint_solve.context.h.device, wp.get_device())
        seen = []
        for step in range(3):
            rigid.step(after_solver_loop=lambda: seen.append(d.qacc.numpy().copy()))
            with self.subTest(step=step):
                self.assertEqual(len(seen), step + 1)
                # the loop of the solve has run when the hook is called
                np.testing.assert_array_equal(seen[-1], d.qacc.numpy())
        rigid.step()
        self.assertEqual(len(seen), 3)

    def test_signature_and_validate_follow_what_the_step_was_created_for(self):
        m, d = step_model("arm")
        rigid = RigidStep(m, d)
        rigid.validate(m, d)
        rigid.step()
        signature = rigid.signature
        hash(signature)
        rigid.step()
        rigid.validate(m, d)
        self.assertEqual(rigid.signature, signature)
        other = clone_data(d)
        self.assertNotEqual(RigidStep(m, other).signature, signature)
        with self.assertRaisesRegex(RuntimeError, "Another rigid model or data; create a new RigidStep"):
            rigid.validate(m, other)
        with self.assertRaisesRegex(RuntimeError, "Another rigid model or data; create a new RigidStep"):
            rigid.validate(step_model("arm")[0], d)
        # an option of the supported configuration; a step checks what validate checks
        flags = m.opt.disableflags
        m.opt.disableflags = flags | DisableBit.WARMSTART
        for check in (lambda: rigid.validate(m, d), rigid.step):
            with self.assertRaisesRegex(RuntimeError, r"changed \(opt\.disableflags\); create a new RigidStep"):
                check()
        m.opt.disableflags = flags
        rigid.validate(m, d)
        # a state array with other memory, and a size
        qpos = d.qpos
        d.qpos = wp.clone(qpos)
        d.njmax += 1
        for check in (lambda: rigid.validate(m, d), rigid.step):
            with self.assertRaisesRegex(RuntimeError, r"changed \(njmax, qpos\); create a new RigidStep"):
                check()
        d.qpos = qpos
        d.njmax -= 1
        rigid.validate(m, d)
        with wp.Tape():
            for check in (lambda: rigid.validate(m, d), rigid.step):
                with self.assertRaisesRegex(RuntimeError, "active tape"):
                    check()
        # the refused steps changed nothing
        reference = clone_data(d)
        rigid.step()
        mjw.step(m, reference)
        self.assertEqual(rigid.signature, signature)
        self._assert_close(_state(d), _state(reference))

    def test_unsupported_configurations_raise(self):
        def euler(m, d):
            m.opt.integrator = mjw.IntegratorType.EULER

        def implicit(m, d):
            m.opt.integrator = mjw.IntegratorType.IMPLICIT

        def sweep_and_prune(m, d):
            m.opt.broadphase = BroadphaseType.SAP_TILE

        def sensor(m, d):
            m.nsensor = 1

        def sensor_callback(m, d):
            m.callback.sensor = lambda m, d, stage: None

        def flex(m, d):
            m.nflex = 1

        def sleeping(m, d):
            m.opt.enableflags |= EnableBit.SLEEP

        def noslip(m, d):
            m.opt.noslip_iterations = 2

        def gradient(m, d):
            d.qvel.requires_grad = True

        def several(m, d):
            euler(m, d)
            flex(m, d)
            noslip(m, d)

        def pyramidal(m, d):
            m.opt.cone = mjw.ConeType.PYRAMIDAL

        def deterministic(m, d):
            m.opt.deterministic = DeterminismType.ATOMICS

        # name, edit of the device model and data, message
        cases = (
            ("Euler integrator", euler, "RigidStep requires the implicitfast integrator$"),
            ("implicit integrator", implicit, "RigidStep requires the implicitfast integrator$"),
            ("sweep-and-prune broadphase", sweep_and_prune, "RigidStep requires the NXN broadphase$"),
            ("sensor", sensor, "RigidStep requires no sensors$"),
            ("sensor callback", sensor_callback, "RigidStep requires no sensor callback$"),
            ("flex", flex, "RigidStep requires no flex$"),
            ("sleeping", sleeping, "RigidStep requires sleeping disabled$"),
            ("noslip iterations", noslip, "RigidStep requires no noslip iterations$"),
            ("state with gradient", gradient, "RigidStep requires state arrays without gradients$"),
            ("several", several, "RigidStep requires the implicitfast integrator, no flex, no noslip iterations$"),
            # the requirements of the parts
            ("pyramidal cones", pyramidal, "NewtonSolve requires elliptic friction cones$"),
            ("deterministic atomics", deterministic, "deterministic atomics$"),
        )
        for name, edit, message in cases:
            m, d = step_model("arm")
            edit(m, d)
            with self.subTest(configuration=name):
                with self.assertRaisesRegex(ValueError, message):
                    RigidStep(m, d)
        m, d = step_model("arm")
        with wp.Tape():
            with self.assertRaisesRegex(ValueError, "RigidStep requires no active tape$"):
                RigidStep(m, d)
        # nothing of a refused step stays installed on the device
        RigidStep(m, d).step()

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_second_step_requests_no_device_memory(self):
        m, d = step_model("gripper")
        rigid = RigidStep(m, d)
        allocator = wp.get_device_allocator(DEVICE)
        requested, signatures = [], []
        with allocation_tracker() as tracker:
            for _ in range(4):
                before = allocations(tracker)
                rigid.step()
                requested.append(allocations(tracker) - before)
                signatures.append(rigid.signature)
        self.assertGreater(rigid.state_scratch.buffer_count, 0)
        self.assertGreater(rigid.scratch.buffer_count, 0)
        self.assertEqual(requested, [rigid.state_scratch.buffer_count + rigid.scratch.buffer_count, 0, 0, 0])
        self.assertEqual(len(set(signatures)), 1)
        self.assertIs(wp.get_device_allocator(DEVICE), allocator)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_captured_steps_replay_without_device_memory(self):
        for name, edit in (("gripper", None), ("arm", _energy), ("boxes", _rne_postconstraint), ("cables", None)):
            m, d = step_model(name, edit=edit)
            rigid = RigidStep(m, d)
            calls = []

            def hook(calls=calls):
                calls.append(None)

            rigid.step(hook)
            signature = rigid.signature
            rng, counts, replayed = np.random.default_rng(5), {}, 0
            with allocation_tracker() as tracker:
                before = allocations(tracker)
                with wp.ScopedCapture() as capture:
                    for _ in range(4):
                        rigid.step(hook)
                captured = allocations(tracker) - before
                for launch in range(6):
                    _kick(d, rng)
                    reference = clone_data(d)
                    for _ in range(4):
                        mjw.step(m, reference)
                    before = allocations(tracker)
                    wp.capture_launch(capture.graph)
                    replayed += allocations(tracker) - before
                    _census(d, counts)
                    self._assert_close(_state(d), _state(reference), model=name, launch=launch)
            with self.subTest(model=name):
                self.assertEqual(captured, 0)
                self.assertEqual(replayed, 0)
                self.assertEqual(rigid.signature, signature)
                # the hook ran for the warm-up and for each captured step, and not in a replay
                self.assertEqual(len(calls), 5)
            self._assert_exercised(counts, name, m, d)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_steps_in_a_conditional_body_capture_and_replay(self):
        # The coupling captures the microsteps of a trial in the body of a graph conditional, which
        # Warp refuses to capture if the body allocates memory. The loop of every constraint solve is
        # a conditional inside that body.
        m, d = step_model("gripper")
        self.assertTrue(m.opt.graph_conditional)
        rigid = RigidStep(m, d)
        rigid.step()
        signature = rigid.signature
        remaining = wp.zeros(1, dtype=int)

        def interval():
            for _ in range(4):
                rigid.step()
            wp.launch(count_down, dim=1, inputs=[remaining])

        rng, counts = np.random.default_rng(9), {}
        with allocation_tracker() as tracker:
            before = allocations(tracker)
            with wp.ScopedCapture() as capture:
                wp.capture_while(remaining, interval)
            captured = allocations(tracker) - before
        for launch in range(4):
            _kick(d, rng)
            reference = clone_data(d)
            for _ in range(8):
                mjw.step(m, reference)
            remaining.fill_(2)
            wp.capture_launch(capture.graph)
            _census(d, counts)
            with self.subTest(launch=launch):
                self.assertEqual(int(remaining.numpy()[0]), 0)
            self._assert_close(_state(d), _state(reference), launch=launch)
        self.assertEqual(captured, 0)
        self.assertEqual(rigid.signature, signature)
        self._assert_exercised(counts, "gripper", m, d)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_capture_needs_an_eager_step(self):
        m, d = step_model("arm")
        with wp.ScopedCapture():
            with self.assertRaisesRegex(RuntimeError, "Create RigidStep outside graph capture"):
                RigidStep(m, d)
        rigid = RigidStep(m, d)
        with self.assertRaisesRegex(RuntimeError, "once outside graph capture"):
            with wp.ScopedCapture():
                rigid.step()
        rigid.step()
        with wp.ScopedCapture():
            rigid.step()

    @unittest.skipUnless(CUDA, "needs a second device")
    def test_step_runs_on_the_device_of_its_data(self):
        m, d = step_model("gripper")
        reference = clone_data(d)
        with wp.ScopedDevice("cpu"):
            rigid = RigidStep(m, d)
            for _ in range(3):
                rigid.step()
        for _ in range(3):
            mjw.step(m, reference)
        self.assertIs(rigid.device, d.qpos.device)
        self._assert_close(_state(d), _state(reference))


if __name__ == "__main__":
    unittest.main()
