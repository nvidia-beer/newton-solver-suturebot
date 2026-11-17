"""Fused body-tree passes against MuJoCo Warp's launch per tree level.

``TreePasses`` and ``mujoco_warp._src.smooth`` get one Model and two equal Data and
must write the same arrays. On the CPU a world is one thread that visits the bodies
in the order of the per-level launches, so every array is compared bit for bit. On
CUDA siblings add to their parent in an unspecified order in both, so the arrays of
a branching tree are compared within the reordering bound of a sum of nbody terms.
A serial chain has one order and is compared bit for bit on CUDA as well.

    PYTHONPATH=tests python -m unittest test_warp_step_tree -v
    NEWTON_TEST_DEVICE=cuda:0 PYTHONPATH=tests python -m unittest test_warp_step_tree -v
"""

import functools
import unittest

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from _warp_step_support import CUDA, DeviceTestCase
from mujoco_warp._src import smooth
from mujoco_warp._src.types import DeterminismType

from newton_solver_suturebot._src.solvers.mujoco.warp_step.tree import TreePasses

EPS = float(np.finfo(np.float32).eps)

# Arrays written by com_pos and crb, and by rne.
POSITION = ("subtree_com", "cinert", "cdof", "crb", "M")
FORCE = ("cacc", "cfrc_int", "qfrc_bias")
# Value of those arrays before the passes run: an entry that a pass leaves out keeps it.
STALE = 7.0

# Every body has one child: each sum up the tree has a single order.
CHAIN = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="hinge" axis="0 1 0" range="-50 50"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <body pos="0.3 0 0">
        <joint type="hinge" axis="0 0 1" range="-70 70"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
        <body pos="0.25 0 0">
          <joint type="slide" axis="1 0 0" range="-0.1 0.1"/>
          <geom type="box" size="0.05 0.03 0.02"/>
          <body pos="0.1 0 0">
            <joint type="ball"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
            <body pos="0.2 0 0">
              <joint type="hinge" axis="1 0 0" range="-90 90"/>
              <geom type="sphere" size="0.05" pos="0 0.05 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# Five levels; the trunk and its first limb have three children each.
TREE = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="hinge" axis="0 0 1" range="-40 40"/>
      <geom type="box" size="0.1 0.08 0.06"/>
      <body pos="0.15 0 0">
        <joint type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
        <body pos="0.2 0.05 0">
          <joint type="hinge" axis="0 1 0" range="-60 60"/>
          <geom type="sphere" size="0.04"/>
        </body>
        <body pos="0.2 -0.05 0">
          <joint type="slide" axis="0 0 1" range="-0.05 0.05"/>
          <geom type="sphere" size="0.03"/>
        </body>
        <body pos="0.2 0 0.05">
          <joint type="ball"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        </body>
      </body>
      <body pos="-0.15 0 0">
        <joint type="hinge" axis="1 0 0" range="-45 45"/>
        <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.03"/>
        <body pos="-0.2 0.04 0">
          <joint type="hinge" axis="0 0 1" range="-30 30"/>
          <geom type="box" size="0.03 0.02 0.05"/>
          <body pos="0 0 -0.1">
            <joint type="hinge" axis="0 1 0" range="-80 80"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.015"/>
          </body>
        </body>
        <body pos="-0.2 -0.04 0">
          <joint type="hinge" axis="0 0 1" range="-30 30"/>
          <geom type="box" size="0.03 0.02 0.05"/>
        </body>
      </body>
      <body pos="0 0 0.12">
        <joint type="ball"/>
        <geom type="sphere" size="0.07"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# An arm that ends in two jaws, and a free body: two unconnected trees below the world body.
ARM_AND_FREE_BODY = """
<mujoco>
  <worldbody>
    <body pos="0.4 0.1 0.6">
      <freejoint/>
      <geom type="box" size="0.03 0.05 0.02"/>
    </body>
    <body pos="0 0 0.2">
      <joint type="hinge" axis="0 0 1" range="-90 90"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04"/>
      <body pos="0 0 0.3">
        <joint type="hinge" axis="0 1 0" range="-80 80"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
        <body pos="0.25 0 0">
          <joint type="hinge" axis="0 1 0" range="-100 100"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025"/>
          <body pos="0.2 0.02 0">
            <joint type="slide" axis="0 1 0" range="0 0.03"/>
            <geom type="box" size="0.03 0.005 0.01"/>
          </body>
          <body pos="0.2 -0.02 0">
            <joint type="slide" axis="0 1 0" range="-0.03 0"/>
            <geom type="box" size="0.03 0.005 0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# One tree level of free bodies.
WIDE = """
<mujoco>
  <worldbody>
    <replicate count="{count}" offset="0.3 0 0">
      <body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
    </replicate>
  </worldbody>
</mujoco>
"""


def _contended_tree(children):
    """Tree whose deepest level spans several warps of a block, all but the first adding to one parent.

    Level 2 holds 32 bodies, one thread each in the first warp of the block. Level 3 gives
    that warp 32 leaves of the first 31 bodies, and the other warps the ``children`` of the
    last, whose additions to one address take their turns. The first warp is done long
    before them, and must still not read that sum, to add it to level 1, until they are.
    """
    leaf = '<body pos="0 {y} -0.1"><joint axis="1 0 0"/><geom type="sphere" size="0.03"/></body>'
    leaves = [2] + [1] * 30 + [children]
    branches = "".join(
        f'<body pos="{0.1 * i} 0 0"><joint axis="0 1 0"/><geom type="sphere" size="0.04"/>'
        + "".join(leaf.format(y=0.05 + 0.01 * k) for k in range(count))
        + "</body>"
        for i, count in enumerate(leaves)
    )
    return f"""
<mujoco>
  <default><joint range="-45 45"/><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body pos="0 0 2"><joint axis="0 0 1"/><geom type="box" size="0.1 0.1 0.1"/>{branches}</body>
  </worldbody>
</mujoco>
"""


MODELS = (
    ("chain", CHAIN),
    ("tree", TREE),
    ("arm and free body", ARM_AND_FREE_BODY),
    ("contended tree", _contended_tree(96)),
    ("widest level", WIDE.format(count=256)),
)


def _state(mjm, worlds, seed=3):
    """Per-world qpos within the joint limits, and qvel."""
    rng = np.random.default_rng(seed)
    qpos = np.tile(mjm.qpos0, (worlds, 1))
    for joint in range(mjm.njnt):
        adr, kind = mjm.jnt_qposadr[joint], mjm.jnt_type[joint]
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            qpos[:, adr : adr + 3] += rng.uniform(-0.2, 0.2, (worlds, 3))
            adr += 3
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            quat = rng.normal(size=(worlds, 4))
            qpos[:, adr : adr + 4] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        else:
            qpos[:, adr] = rng.uniform(*mjm.jnt_range[joint], worlds)
    return qpos, rng.uniform(-1.0, 1.0, (worlds, mjm.nv))


def _model(mjm, worlds):
    """Device model; several worlds get their own body masses, so that a world's row of a model array matters."""
    m = mjw.put_model(mjm)
    if worlds > 1:
        mass = mjm.body_mass * np.random.default_rng(13).uniform(0.5, 1.5, (worlds, mjm.nbody))
        subtree = mass.copy()
        for body in range(mjm.nbody - 1, 0, -1):
            subtree[:, mjm.body_parentid[body]] += subtree[:, body]
        m.body_mass = wp.array(mass, dtype=float)
        m.body_subtreemass = wp.array(subtree, dtype=float)
    return m


def _load(m, d, state):
    """Put ``state`` into ``d``, run MuJoCo Warp's kinematics and leave stale values where the tree passes write."""
    for name, value in zip(("qpos", "qvel"), state, strict=True):
        getattr(d, name).assign(value.astype(np.float32))
    smooth.kinematics(m, d)
    for name in POSITION + FORCE:
        getattr(d, name).fill_(STALE)


def _data(mjm, m, worlds, state):
    """Data of ``worlds`` worlds, prepared by ``_load``."""
    d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=worlds)
    _load(m, d, state)
    return d


def _tree_passes(m, d, passes=None):
    """Arrays that the tree passes write in step order, with ``passes`` or with MuJoCo Warp's launch per level."""
    if passes:
        com_pos, crb, rne = passes.com_pos, passes.crb, passes.rne
    else:
        com_pos, crb, rne = (functools.partial(function, m, d) for function in (smooth.com_pos, smooth.crb, smooth.rne))
    com_pos()
    crb()
    smooth.com_vel(m, d)
    rne()
    return {name: getattr(d, name).numpy().copy() for name in POSITION + FORCE}


def _scale(arrays, field):
    """Magnitude of the terms that the entries of an array sum.

    qfrc_bias projects the internal forces on the joint axes and can be far smaller than
    both, so its terms have the size of an axis' 1-norm times a force.
    """
    if field == "qfrc_bias":
        return float(np.abs(arrays["cdof"]).sum(axis=-1).max() * np.abs(arrays["cfrc_int"]).max())
    return float(np.abs(arrays[field]).max())


class TreePassesTests(DeviceTestCase):
    def _assert_same(self, actual, expected, field, terms, ordered):
        if ordered:
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
            return
        # Reordering a sum of n terms moves it by at most 2 (n - 1) eps times the sum of their magnitudes.
        bound = 2.0 * terms * EPS * _scale(expected, field)
        np.testing.assert_allclose(actual[field], expected[field], rtol=0.0, atol=bound, err_msg=field)

    def test_passes_match_one_launch_per_level(self):
        scalar = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        for name, xml in MODELS:
            mjm = mujoco.MjModel.from_xml_string(xml)
            self.assertTrue(mjm.jnt_limited[np.isin(mjm.jnt_type, scalar)].all(), name)
            # one child per body leaves the additions of a pass a single order, on any device
            ordered = not wp.get_device().is_cuda or int(np.bincount(mjm.body_parentid[1:]).max()) == 1
            for worlds in (1, 3):
                m = _model(mjm, worlds)
                state = _state(mjm, worlds)
                expected = _tree_passes(m, _data(mjm, m, worlds, state))
                d = _data(mjm, m, worlds, state)
                actual = _tree_passes(m, d, TreePasses(m, d))
                for field, value in expected.items():
                    with self.subTest(model=name, worlds=worlds, field=field):
                        self.assertEqual(value.shape[0], worlds)
                        self.assertTrue(np.isfinite(value).all())
                        self.assertGreater(float(np.abs(value).max()), 0.0)
                        self._assert_same(actual, expected, field, mjm.nbody, ordered)

    @unittest.skipUnless(CUDA, "the threads of a world share a block on CUDA only")
    def test_contended_level_is_complete_before_the_next(self):
        # Without the barrier after a level the first warp of a block passes a sum on while the
        # other warps still add to it. That is a race, so the passes run repeatedly.
        mjm = mujoco.MjModel.from_xml_string(_contended_tree(96))
        m = _model(mjm, 2)
        state = _state(mjm, 2)
        for repeat in range(8):
            expected = _tree_passes(m, _data(mjm, m, 2, state))
            d = _data(mjm, m, 2, state)
            actual = _tree_passes(m, d, TreePasses(m, d))
            for field in expected:
                with self.subTest(repeat=repeat, field=field):
                    self._assert_same(actual, expected, field, mjm.nbody, ordered=False)

    @unittest.skipUnless(CUDA, "CUDA graph capture")
    def test_captured_passes_replay_on_a_new_state(self):
        mjm = mujoco.MjModel.from_xml_string(ARM_AND_FREE_BODY)
        m = _model(mjm, 3)
        d = _data(mjm, m, 3, _state(mjm, 3))
        passes = TreePasses(m, d)
        # run eagerly first, as the solver does before it captures: the kernel modules load here
        _tree_passes(m, d, passes)
        with wp.ScopedCapture() as capture:
            passes.com_pos()
            passes.crb()
            smooth.com_vel(m, d)
            passes.rne()
        state = _state(mjm, 3, seed=4)
        expected = _tree_passes(m, _data(mjm, m, 3, state))
        _load(m, d, state)
        wp.capture_launch(capture.graph)
        actual = {field: getattr(d, field).numpy() for field in POSITION + FORCE}
        for field in actual:
            with self.subTest(field=field):
                self._assert_same(actual, expected, field, mjm.nbody, ordered=False)

    @unittest.skipUnless(CUDA, "needs a second device")
    def test_passes_run_on_the_device_of_their_data(self):
        mjm = mujoco.MjModel.from_xml_string(TREE)
        m = _model(mjm, 2)
        state = _state(mjm, 2)
        expected = _tree_passes(m, _data(mjm, m, 2, state))
        d = _data(mjm, m, 2, state)
        with wp.ScopedDevice("cpu"):
            passes = TreePasses(m, d)
            passes.com_pos()
            passes.crb()
        smooth.com_vel(m, d)
        with wp.ScopedDevice("cpu"):
            passes.rne()
        self.assertIs(passes.device, d.subtree_com.device)
        actual = {field: getattr(d, field).numpy() for field in POSITION + FORCE}
        for field in actual:
            with self.subTest(field=field):
                self._assert_same(actual, expected, field, mjm.nbody, ordered=False)

    def test_level_tables_and_block_size(self):
        for name, xml in MODELS:
            with self.subTest(model=name):
                mjm = mujoco.MjModel.from_xml_string(xml)
                m = _model(mjm, 1)
                passes = TreePasses(m, _data(mjm, m, 1, _state(mjm, 1)))
                depth = np.zeros(mjm.nbody, dtype=int)
                for body in range(1, mjm.nbody):
                    depth[body] = depth[mjm.body_parentid[body]] + 1
                levels, start = passes.body_levels.numpy(), passes.body_level_start.numpy()
                self.assertEqual(passes.nlevel, depth.max() + 1)
                # every level in turn, its bodies in ascending order as in the model's body_tree
                np.testing.assert_array_equal(levels, np.argsort(depth, kind="stable"))
                np.testing.assert_array_equal(start, np.cumsum([0, *np.bincount(depth)]))
                np.testing.assert_array_equal(levels, np.concatenate([level.numpy() for level in m.body_tree]))
                for array in (passes.body_levels, passes.body_level_start):
                    self.assertEqual(array.device, wp.get_device())
                    self.assertEqual(array.dtype, wp.int32)
                # whole warps that hold one thread per body of the widest level
                width = int(np.bincount(depth).max())
                self.assertEqual(passes.block_dim, 32 * ((width + 31) // 32))

    def test_deterministic_atomics_are_rejected(self):
        mjm = mujoco.MjModel.from_xml_string(TREE)
        m = _model(mjm, 1)
        d = _data(mjm, m, 1, _state(mjm, 1))
        for flags in (DeterminismType.ATOMICS, DeterminismType.ALL):
            m.opt.deterministic = flags
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                TreePasses(m, d)
        m.opt.deterministic = DeterminismType.CONTACTS | DeterminismType.CONSTRAINT
        TreePasses(m, d)

    def test_level_wider_than_a_block_is_rejected(self):
        mjm = mujoco.MjModel.from_xml_string(WIDE.format(count=257))
        m = _model(mjm, 1)
        self.assertEqual([level.size for level in m.body_tree], [1, 257])
        with self.assertRaises(ValueError):
            TreePasses(m, _data(mjm, m, 1, _state(mjm, 1)))


if __name__ == "__main__":
    unittest.main()
