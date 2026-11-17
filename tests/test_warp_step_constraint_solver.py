"""``NewtonSolve`` against the constraint solve of MuJoCo Warp.

``NewtonSolve.solve`` and ``mujoco_warp._src.solver.solve`` get one Model and two equal
Data and must write the same arrays. On the CPU both run the same operations in the same
order, so every array is compared bit for bit.

On CUDA the contacts of one kinematic tree add their cone curvature to the same Hessian
entries in an unspecified order, in both solves, so MuJoCo Warp's solve does not repeat
itself bit for bit there: over 1000 states, two of its solves from equal Data differ by up
to 1e-5 (``boxes``) and 2e-5 (``humanoid``) of the largest entry. These models are
compared to 1e-3 of the largest entry on CUDA, and their constraint states and iteration
counts are not compared. In ``singles`` every tree has at most one contact, which leaves
one order, and it is compared bit for bit on CUDA as well. Its worlds have more than 32
constraint rows, which several warps of a block update.

No test here fails without the block barriers of the fused kernels. On the CPU a block is
one thread. On CUDA the degrees of freedom of a world, at most 32, are on the first warp
of its block. That warp reads the row forces of the other warps after it has updated 32
rows of its own and summed over them, and in 1000 states of each model it never read one
before it was written.
"""

import dataclasses
import unittest
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from _warp_step_support import (
    BOXES,
    CHAIN,
    CUDA,
    FLOOR_SCENE,
    HUMANOID,
    DeviceTestCase,
    allocation_tracker,
    allocations,
    clone_data,
    count_down,
    data_arrays,
    scene_options,
)
from mujoco_warp._src import solver, support
from mujoco_warp._src.types import (
    ConstraintState,
    ConstraintType,
    DeterminismType,
    EnableBit,
    IntegratorType,
    OverflowType,
)

from newton_solver_suturebot._src.solvers.mujoco.warp_step.constraint_solver import NewtonSolve

# nv 31. Two free bodies and five arms on ball joints, each resting on the plane with one contact
# of 1, 3, 4 or 6 constraint rows.
ARM = """
    <body pos="{x} -0.6 0.15">
      <joint type="ball"/><geom type="sphere" size="0.05" pos="0.17 0 -0.101" condim="{condim}"/>
    </body>
"""
SINGLES = FLOOR_SCENE.format(
    chain=CHAIN,
    bodies="""
    <body pos="0 0 0.05"><freejoint/><geom type="sphere" size="0.05" condim="6"/></body>
    <body pos="0.4 0 0.06"><freejoint/><geom type="ellipsoid" size="0.06 0.05 0.06" condim="3"/></body>
"""
    + "".join(ARM.format(x=0.5 * index, condim=condim) for index, condim in enumerate((6, 6, 6, 4, 1))),
)

# A hinge without limit or contact: no constraint rows.
PENDULUM = """
<mujoco>
  <option integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <worldbody>
    <body pos="0 0 1">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# name: XML text or file, constraint rows of a world
MODELS = {"boxes": (BOXES, 96), "singles": (SINGLES, 96), "humanoid": (HUMANOID, 128)}

# name, worlds, warm start
CASES = (
    ("boxes", 2, True),
    ("boxes", 1, False),
    ("singles", 3, True),
    ("singles", 1, False),
    ("humanoid", 2, True),
)

# Integration state that a step advances.
STATE = ("qpos", "qvel", "qacc_warmstart", "time")


def _native(source, *, warmstart=True, edit=None):
    """MuJoCo model and data of an XML text or file."""
    if isinstance(source, Path):
        mjm = mujoco.MjModel.from_xml_path(str(source))
    else:
        mjm = mujoco.MjModel.from_xml_string(source)
    if not warmstart:
        mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    if edit:
        edit(mjm)
    mjd = mujoco.MjData(mjm)
    if mjm.nkey:
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    return mjm, mjd


def _kick(d, rng):
    """Random velocities, so that contacts slide and a solve needs iterations."""
    d.qvel.assign(rng.uniform(-0.5, 0.5, d.qvel.shape).astype(np.float32))


def _model(name, worlds, *, warmstart=True, steps=20):
    """Device model after ``steps`` steps of mujoco_warp.step, which bring contacts and limits into play."""
    source, njmax = MODELS[name]
    mjm, mjd = _native(source, warmstart=warmstart, edit=scene_options if isinstance(source, Path) else None)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=worlds, njmax=njmax)
    # distinct worlds, so that a per-world indexing error cannot cancel
    _kick(d, np.random.default_rng(5))
    for _ in range(steps):
        mjw.step(m, d)
    return m, d


def _exact(name):
    """Whether two solves of this model from equal Data are equal bit for bit on the test device."""
    return not CUDA or name == "singles"


def _host(array):
    """Host copy of an array. On the CPU ``numpy()`` is a view of the array, which later writes change."""
    return array.numpy().copy()


def _smooth(m, d):
    """The pieces of mujoco_warp.forward before the constraint solve, without sensors and energy."""
    mjw.fwd_position(m, d, factorize=False)
    mjw.fwd_velocity(m, d)
    mjw.fwd_actuation(m, d)
    mjw.fwd_acceleration(m, d, factorize=True)


def _active(d):
    return np.arange(d.efc.force.shape[1])[None, :] < d.nefc.numpy()[:, None]


def _solved(d):
    """Arrays that a solve writes; it leaves the constraint rows past nefc as they were."""
    active = _active(d)
    return {
        "qacc": _host(d.qacc),
        "qfrc_constraint": _host(d.qfrc_constraint),
        "efc.Ma": _host(d.efc.Ma),
        "efc.force": np.where(active, d.efc.force.numpy(), 0.0),
        "efc.state": np.where(active, d.efc.state.numpy(), 0),
        "solver_niter": _host(d.solver_niter),
        "overflow": _host(d.overflow),
    }


def _state(d):
    return {field: _host(getattr(d, field)) for field in STATE}


def _context(context, d):
    """Context entries that both solves have written when they end.

    Left out are mv, where MuJoCo Warp leaves the product of the search before the last, and
    quad, which the linesearch writes for the rows of contacts only.
    """
    active = _active(d)[:, : d.njmax]
    fields = (
        "h",
        "grad",
        "search",
        "search_dot",
        "newton_decrement",
        "alpha",
        "improvement",
        "inertia_scale",
        "grad_scale",
        "done",
        "search_unchanged",
    )
    return {
        **{name: _host(getattr(context, name)) for name in fields},
        "Jaref": np.where(active, context.Jaref.numpy(), 0.0),
        "jv": np.where(active, context.jv.numpy(), 0.0),
    }


def _poison(context):
    """Overwrite every array of a solver context: NaN, -7 and True."""
    for field in dataclasses.fields(context):
        array = getattr(context, field.name)
        if not isinstance(array, wp.array) or array.size == 0:
            continue
        if array.dtype is wp.bool:
            array.fill_(True)
        elif array.dtype is wp.int32:
            array.fill_(-7)
        else:
            array.fill_(float("nan"))


def _contacts_per_tree(m, d):
    """Largest number of contacts with constraint rows that one kinematic tree of one world takes part in."""
    count = int(d.nacon.numpy()[0])
    rows = d.contact.efc_address.numpy()[:count, 0] >= 0
    geoms = d.contact.geom.numpy()[:count][rows]
    worlds = d.contact.worldid.numpy()[:count][rows]
    roots = m.body_rootid.numpy()[m.geom_bodyid.numpy()[geoms]]
    counts = {}
    for world, pair in zip(worlds, roots, strict=True):
        # root 0 is the world body
        for root in set(pair.tolist()) - {0}:
            counts[world, root] = counts.get((world, root), 0) + 1
    return max(counts.values(), default=0)


def _census(m, d, counts):
    """Largest number of rows of each kind over the solved states."""
    active, state, kind = _active(d), d.efc.state.numpy(), d.efc.type.numpy()
    found = {
        "rows": int(d.nefc.numpy().max()),
        "equality": int(d.ne.numpy().max()),
        "friction loss": int(d.nf.numpy().max()),
        "limit": int((active & (kind == ConstraintType.LIMIT_JOINT)).sum(axis=1).max()),
        "cone": int((active & (state == ConstraintState.CONE)).sum(axis=1).max()),
        "iterations": int(d.solver_niter.numpy().max()),
        "contacts of a tree": _contacts_per_tree(m, d),
    }
    for name, value in found.items():
        counts[name] = max(counts.get(name, 0), value)


class NewtonSolveTests(DeviceTestCase):
    def _assert_same(self, actual, expected, exact, **context):
        for name, value in expected.items():
            with self.subTest(field=name, **context):
                self.assertTrue(np.isfinite(value).all())
                if exact:
                    np.testing.assert_array_equal(actual[name], value, err_msg=name)
                elif value.dtype.kind == "f":
                    bound = 1e-3 * float(np.abs(value).max())
                    np.testing.assert_allclose(actual[name], value, rtol=0.0, atol=bound, err_msg=name)

    def _assert_exercised(self, counts, name, d, **context):
        with self.subTest(model=name, **context):
            self.assertLessEqual(counts["rows"], d.njmax)
            self.assertGreater(counts["limit"], 0)
            self.assertGreater(counts["cone"], 0)
            self.assertGreater(counts["iterations"], 2)
            if name != "humanoid":
                self.assertGreater(counts["equality"], 0)
                self.assertGreater(counts["friction loss"], 0)
            if name == "singles":
                self.assertEqual(counts["contacts of a tree"], 1)
                # the rows of a world span more than one warp of its block
                self.assertGreater(counts["rows"], 32)
            else:
                self.assertGreater(counts["contacts of a tree"], 1)

    def test_solve_matches_mujoco_warp(self):
        for name, worlds, warmstart in CASES:
            m, d = _model(name, worlds, warmstart=warmstart)
            newton = NewtonSolve(m, d)
            rng, counts = np.random.default_rng(11), {}
            for state in range(24):
                if state % 4 == 0:
                    _kick(d, rng)
                _smooth(m, d)
                reference = clone_data(d)
                solver.solve(m, reference)
                newton.solve()
                _census(m, reference, counts)
                context = {"model": name, "warmstart": warmstart, "state": state}
                self._assert_same(_solved(d), _solved(reference), _exact(name), **context)
                mjw.implicit(m, d)
            self._assert_exercised(counts, name, d, warmstart=warmstart)

    def test_solve_changes_the_data_as_mujoco_warp_does(self):
        # Every array of Data: the solve writes what MuJoCo Warp's solve writes, and nothing else.
        written = {"d.qacc", "d.qfrc_constraint", "d.efc.Ma", "d.efc.force", "d.efc.state", "d.solver_niter"}
        for name, worlds, warmstart in CASES:
            if not _exact(name):
                continue
            m, d = _model(name, worlds, warmstart=warmstart)
            newton = NewtonSolve(m, d)
            rng = np.random.default_rng(31)
            for state in range(3):
                _kick(d, rng)
                _smooth(m, d)
                # values that no solve leaves, in the arrays that a solve writes before it reads them
                for array in (d.qacc, d.qfrc_constraint, d.efc.Ma, d.efc.force):
                    array.fill_(float("nan"))
                d.efc.state.fill_(-1)
                d.solver_niter.fill_(-1)
                before = {field: _host(array) for field, array in data_arrays(d)}
                reference = clone_data(d)
                solver.solve(m, reference)
                newton.solve()
                context = {"model": name, "warmstart": warmstart, "state": state}
                changed = set()
                for (field, actual), (_, expected) in zip(data_arrays(d), data_arrays(reference), strict=True):
                    with self.subTest(field=field, **context):
                        np.testing.assert_array_equal(actual.numpy(), expected.numpy(), err_msg=field)
                    if not np.array_equal(expected.numpy(), before[field], equal_nan=True):
                        changed.add(field)
                with self.subTest(**context):
                    self.assertGreater(len(before), 100)
                    self.assertLessEqual(written, changed)
                    self.assertLessEqual(changed, written | {"d.overflow"})
                # the rows of a world are finite, and the rows past them are as they were
                self._assert_same(_solved(d), _solved(reference), True, **context)
                mjw.implicit(m, d)

    def test_context_matches_mujoco_warp(self):
        # The context after a solve, as MuJoCo Warp's own solve leaves the one it is given: the Hessian,
        # which the contact metric of the coupling reads, and what the termination test reads.
        cuda = wp.get_device().is_cuda
        for name, worlds, warmstart in CASES:
            m, d = _model(name, worlds, warmstart=warmstart)
            newton = NewtonSolve(m, d)
            product = wp.zeros_like(newton.context.mv)
            rng = np.random.default_rng(13)
            for state in range(8):
                if state % 2 == 0:
                    _kick(d, rng)
                _smooth(m, d)
                reference = clone_data(d)
                expected = solver._create_solver_context(m, reference)
                solver._solve(m, reference, expected)
                newton.solve()
                context = {"model": name, "warmstart": warmstart, "state": state}
                with self.subTest(**context):
                    self.assertIs(newton.context.h.device, d.qacc.device)
                    self.assertEqual(newton.context.h.shape, (worlds, m.nv_pad, m.nv_pad))
                    self.assertTrue(newton.context.done.numpy().all())
                if _exact(name):
                    self._assert_same(_context(newton.context, d), _context(expected, reference), True, **context)
                    # the degrees of freedom of a world add to its grad_dot, on CUDA in an unspecified order
                    with self.subTest(field="grad_dot", **context):
                        grad_dot = newton.context.grad_dot.numpy()
                        np.testing.assert_allclose(grad_dot, expected.grad_dot.numpy(), rtol=1e-4 if cuda else 0.0)
                else:
                    hessian = {"h": newton.context.h.numpy()}
                    self._assert_same(hessian, {"h": expected.h.numpy()}, False, **context)
                # mv is the product of the search that the solve ended with
                support.mul_m(m, d, product, newton.context.search)
                with self.subTest(field="mv", **context):
                    np.testing.assert_array_equal(newton.context.mv.numpy(), product.numpy())
                mjw.implicit(m, d)

    def test_twenty_steps_on_one_context(self):
        # One NewtonSolve solves 20 consecutive steps, with MuJoCo Warp's forward pieces around the
        # solve. Each solve is compared with a MuJoCo Warp solve, on a new context, from equal Data.
        # On the CPU two more rollouts must stay equal to it bit for bit: the same pieces around
        # MuJoCo Warp's solve, and mujoco_warp.step alone. On CUDA two rollouts differ in MuJoCo
        # Warp itself, where contacts and constraint rows take their slots from races between
        # threads: after these 20 steps two of its humanoid rollouts are up to 3e-4 of the largest
        # entry apart.
        cpu = not wp.get_device().is_cuda
        for name, worlds, warmstart in CASES:
            m, d = _model(name, worlds, warmstart=warmstart)
            pieces, stepped = (clone_data(d), clone_data(d)) if cpu else (None, None)
            newton = NewtonSolve(m, d)
            context = newton.context
            rng, counts = np.random.default_rng(17), {}
            for step in range(20):
                where = {"model": name, "warmstart": warmstart, "step": step}
                if step % 10 == 0:
                    _kick(d, rng)
                    if cpu:
                        wp.copy(pieces.qvel, d.qvel)
                        wp.copy(stepped.qvel, d.qvel)
                _smooth(m, d)
                reference = clone_data(d)
                solver.solve(m, reference)
                newton.solve()
                _census(m, reference, counts)
                self._assert_same(_solved(d), _solved(reference), _exact(name), **where)
                mjw.implicit(m, d)
                if cpu:
                    _smooth(m, pieces)
                    solver.solve(m, pieces)
                    mjw.implicit(m, pieces)
                    mjw.step(m, stepped)
                    actual = {**_solved(d), **_state(d)}
                    self._assert_same(actual, {**_solved(pieces), **_state(pieces)}, True, rollout="pieces", **where)
                    self._assert_same(actual, {**_solved(stepped), **_state(stepped)}, True, rollout="step", **where)
            self.assertIs(newton.context, context)
            self._assert_exercised(counts, name, d, warmstart=warmstart)

    def test_solve_reads_nothing_that_an_earlier_solve_left(self):
        # The context is not zeroed between solves: every entry that a solve reads, it has written.
        for name, worlds, warmstart in CASES:
            m, d = _model(name, worlds, warmstart=warmstart)
            newton = NewtonSolve(m, d)
            rng = np.random.default_rng(37)
            for state in range(4):
                _kick(d, rng)
                _smooth(m, d)
                reference = clone_data(d)
                solver.solve(m, reference)
                _poison(newton.context)
                newton.solve()
                context = {"model": name, "warmstart": warmstart, "state": state}
                self._assert_same(_solved(d), _solved(reference), _exact(name), **context)
                with self.subTest(**context):
                    self.assertTrue(np.isfinite(newton.context.h.numpy()).all())
                mjw.implicit(m, d)

    def test_iteration_limit(self):
        # Worlds that reach opt.iterations unconverged end the loop and report the overflow.
        for conditional in (True, False):
            m, d = _model("singles", 3)
            m.opt.graph_conditional = conditional
            m.opt.iterations = 2
            m.opt.warn_overflow = 0
            newton = NewtonSolve(m, d)
            rng = np.random.default_rng(41)
            limited = 0
            for state in range(8):
                _kick(d, rng)
                d.overflow.zero_()
                _smooth(m, d)
                reference = clone_data(d)
                solver.solve(m, reference)
                newton.solve()
                self._assert_same(_solved(d), _solved(reference), True, conditional=conditional, state=state)
                overflow = reference.overflow.numpy() & int(OverflowType.ITERATIONS)
                limited += int((overflow != 0).sum())
                with self.subTest(conditional=conditional, state=state):
                    self.assertLessEqual(int(reference.solver_niter.numpy().max()), 2)
                    self.assertTrue((reference.solver_niter.numpy()[overflow != 0] == 2).all())
                mjw.implicit(m, d)
            self.assertGreater(limited, 0)

    def test_after_loop_is_called_once_per_solve(self):
        # the loop as a graph conditional, as a loop in Python, and without iterations
        for conditional, iterations in ((True, 100), (False, 100), (True, 0), (False, 0)):
            m, d = _model("singles", 2)
            m.opt.graph_conditional = conditional
            m.opt.iterations = iterations
            newton = NewtonSolve(m, d)
            rng = np.random.default_rng(19)
            for state in range(3):
                _kick(d, rng)
                _smooth(m, d)
                reference = clone_data(d)
                solver.solve(m, reference)
                seen = []
                newton.solve(after_loop=lambda seen=seen, d=d: seen.append(_solved(d)))
                context = {"conditional": conditional, "iterations": iterations, "state": state}
                with self.subTest(**context):
                    self.assertEqual(len(seen), 1)
                    self.assertEqual(int(d.solver_niter.numpy().max()) > 0, iterations > 0)
                # the loop has run when the hook is called, and the solve writes nothing after it
                self._assert_same(seen[0], _solved(d), True, hook=True, **context)
                self._assert_same(_solved(d), _solved(reference), True, **context)
                mjw.implicit(m, d)

    def test_model_without_constraint_rows(self):
        mjm, mjd = _native(PENDULUM)
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=2, njmax=0)
        _kick(d, np.random.default_rng(23))
        newton = NewtonSolve(m, d)
        for state in range(3):
            _smooth(m, d)
            reference = clone_data(d)
            solver.solve(m, reference)
            d.qacc.zero_()
            d.solver_niter.fill_(7)
            calls = []
            newton.solve(after_loop=lambda calls=calls: calls.append(None))
            with self.subTest(state=state):
                self.assertEqual(len(calls), 1)
                self.assertTrue(np.abs(reference.qacc.numpy()).min() > 0.0)
                np.testing.assert_array_equal(d.qacc.numpy(), d.qacc_smooth.numpy())
                np.testing.assert_array_equal(d.qacc.numpy(), reference.qacc.numpy())
                np.testing.assert_array_equal(d.solver_niter.numpy(), reference.solver_niter.numpy())
            mjw.implicit(m, d)

    def test_unsupported_configurations_raise(self):
        def pyramidal(mjm):
            mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        def conjugate_gradient(mjm):
            mjm.opt.solver = mujoco.mjtSolver.mjSOL_CG

        def sparse(mjm):
            mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE

        def noslip(mjm):
            mjm.opt.noslip_iterations = 2

        def sleeping(m):
            m.opt.enableflags |= EnableBit.SLEEP

        def discrete(m):
            m.opt.integrator = IntegratorType.DISCRETE

        def deterministic(m):
            m.opt.deterministic = DeterminismType.ATOMICS

        free = '<body pos="{x} 0 0.05"><freejoint/><geom type="sphere" size="0.05"/></body>'
        wide = FLOOR_SCENE.format(chain=CHAIN, bodies="".join(free.format(x=0.3 * i) for i in range(5)))
        # name, xml, edit of the MuJoCo model, edit of the device model, njmax, message
        cases = (
            ("pyramidal cones", SINGLES, pyramidal, None, 96, "elliptic friction cones"),
            ("CG solver", SINGLES, conjugate_gradient, None, 96, "the Newton solver"),
            ("sparse Jacobian", SINGLES, sparse, None, 96, "a dense constraint Jacobian"),
            ("noslip iterations", SINGLES, noslip, None, 96, "no noslip iterations"),
            ("sleeping", SINGLES, None, sleeping, 96, "sleeping disabled"),
            ("DISCRETE integrator", SINGLES, None, discrete, 96, "an integrator other than DISCRETE"),
            ("deterministic atomics", SINGLES, None, deterministic, 96, "atomics that are not deterministic"),
            ("34 degrees of freedom", wide, None, None, 96, "at most 32 degrees of freedom"),
            ("257 constraint rows", SINGLES, None, None, 257, "at most 256 constraint rows per world"),
        )
        for name, xml, edit_native, edit_device, njmax, message in cases:
            mjm, mjd = _native(xml, edit=edit_native)
            m = mjw.put_model(mjm)
            d = mjw.put_data(mjm, mjd, njmax=njmax)
            if edit_device:
                edit_device(m)
            with self.subTest(configuration=name):
                with self.assertRaisesRegex(ValueError, f"NewtonSolve requires {message}$"):
                    NewtonSolve(m, d)
        mjm, mjd = _native(SINGLES)
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, njmax=256)
        # the largest supported data
        self.assertEqual(NewtonSolve(m, d).context.h.shape, (1, m.nv_pad, m.nv_pad))
        compacted = dataclasses.replace(d, M=wp.zeros((d.nworld, m.nv, m.nv), dtype=float))
        with self.assertRaisesRegex(ValueError, "NewtonSolve requires an inertia matrix that is not compacted$"):
            NewtonSolve(m, compacted)
        m.opt.cone = mjw.ConeType.PYRAMIDAL
        m.opt.solver = mjw.SolverType.CG
        with self.assertRaisesRegex(ValueError, "NewtonSolve requires the Newton solver, elliptic friction cones$"):
            NewtonSolve(m, d)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_captured_solve_replays_eager_solves(self):
        for name, worlds, warmstart in CASES:
            m, d = _model(name, worlds, warmstart=warmstart)
            self.assertTrue(m.opt.graph_conditional)
            newton = NewtonSolve(m, d)
            _smooth(m, d)
            newton.solve()
            calls = []
            with wp.ScopedCapture() as capture:
                newton.solve(after_loop=lambda calls=calls: calls.append(None))
            self.assertEqual(len(calls), 1)
            rng, counts = np.random.default_rng(29), {}
            for state in range(12):
                if state % 4 == 0:
                    _kick(d, rng)
                _smooth(m, d)
                reference, eager = clone_data(d), clone_data(d)
                solver.solve(m, reference)
                NewtonSolve(m, eager).solve()
                wp.capture_launch(capture.graph)
                _census(m, reference, counts)
                context = {"model": name, "warmstart": warmstart, "state": state}
                self._assert_same(_solved(d), _solved(reference), _exact(name), against="mujoco_warp", **context)
                self._assert_same(_solved(d), _solved(eager), _exact(name), against="eager", **context)
                mjw.implicit(m, d)
            self.assertEqual(len(calls), 1)
            self._assert_exercised(counts, name, d, warmstart=warmstart)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_solve_in_the_body_of_a_graph_conditional(self):
        # The coupling captures the rigid step in the body of a conditional node, which Warp
        # refuses to capture if the body allocates memory.
        m, d = _model("singles", 2)
        newton = NewtonSolve(m, d)
        _smooth(m, d)
        newton.solve()
        remaining = wp.zeros(1, dtype=int)

        def body():
            newton.solve()
            wp.launch(count_down, dim=1, outputs=[remaining])

        with wp.ScopedCapture() as capture:
            wp.capture_while(remaining, while_body=body)
        rng = np.random.default_rng(43)
        for state in range(6):
            _kick(d, rng)
            _smooth(m, d)
            reference = clone_data(d)
            solver.solve(m, reference)
            d.qacc.zero_()
            # the second pass of the body solves again from the same inputs
            remaining.fill_(2)
            wp.capture_launch(capture.graph)
            with self.subTest(state=state):
                self.assertEqual(int(remaining.numpy()[0]), 0)
                self.assertGreater(int(reference.solver_niter.numpy().max()), 1)
            self._assert_same(_solved(d), _solved(reference), True, state=state)
            mjw.implicit(m, d)

    @unittest.skipUnless(CUDA, "Warp installs custom allocators on CUDA devices only")
    def test_solve_creates_no_device_array(self):
        m, d = _model("boxes", 1)
        newton = NewtonSolve(m, d)
        _smooth(m, d)
        newton.solve()
        reference = clone_data(d)
        with allocation_tracker() as tracker:
            before = allocations(tracker)
            newton.solve()
            kept = allocations(tracker) - before
            # MuJoCo Warp's solve creates its context, which shows that the count sees such arrays
            before = allocations(tracker)
            solver.solve(m, reference)
            created = allocations(tracker) - before
        self.assertEqual(kept, 0)
        self.assertGreater(created, 0)

    @unittest.skipUnless(CUDA, "needs a second device")
    def test_solve_runs_on_the_device_of_its_data(self):
        m, d = _model("singles", 1)
        _smooth(m, d)
        reference = clone_data(d)
        solver.solve(m, reference)
        with wp.ScopedDevice("cpu"):
            newton = NewtonSolve(m, d)
            newton.solve()
        self.assertIs(newton.context.h.device, d.qacc.device)
        self._assert_same(_solved(d), _solved(reference), True)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_creation_during_capture_raises(self):
        m, d = _model("singles", 1, steps=1)
        with wp.ScopedCapture():
            with self.assertRaisesRegex(RuntimeError, "outside graph capture"):
                NewtonSolve(m, d)


if __name__ == "__main__":
    unittest.main()
