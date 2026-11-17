"""Shared pieces of the rigid step tests.

The test device, copies of a MuJoCo Warp Data, counters of device allocations and device
activities, and the scenes that more than one of ``test_warp_step*.py`` and
``test_mjw_fused_launches.py`` step.
"""

import collections
import dataclasses
import io
import os
import re
import unittest
from contextlib import contextmanager
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
CUDA = wp.get_device(DEVICE).is_cuda


class DeviceTestCase(unittest.TestCase):
    """Test case that runs with the test device as the default Warp device."""

    def setUp(self):
        self.enterContext(wp.ScopedDevice(DEVICE))


def clone_data(value):
    """Copy of a MuJoCo Warp Data with arrays of its own."""
    if isinstance(value, wp.array):
        return wp.clone(value)
    if dataclasses.is_dataclass(value):
        fields = {
            field.name: clone_data(getattr(value, field.name)) for field in dataclasses.fields(value) if field.init
        }
        return dataclasses.replace(value, **fields)
    if isinstance(value, (tuple, list)):
        return type(value)(clone_data(item) for item in value)
    return value


def data_arrays(value, name="d"):
    """Every array of a MuJoCo Warp Data, by name."""
    if isinstance(value, wp.array):
        yield name, value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from data_arrays(getattr(value, field.name), f"{name}.{field.name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from data_arrays(item, f"{name}[{index}]")


@wp.kernel(enable_backward=False)
def count_down(remaining: wp.array[int]):
    remaining[0] = remaining[0] - 1


@contextmanager
def allocation_tracker():
    """Warp's allocation tracker without the records of earlier trackers.

    Such a record stays after its memory is freed, and an allocation that gets the same
    address is then missing from the live count.
    """
    with wp.ScopedMemoryTracker("scratch", print=False) as tracker:
        tracker.clear()
        yield tracker


def allocations(tracker, section="Total"):
    """Allocations of the test device in a section, "Total" or "Live", of Warp's allocation report."""
    report = io.StringIO()
    tracker.report(file=report)
    lines = report.getvalue().splitlines()
    for line in lines[lines.index(f"  {section} allocations:") + 1 :]:
        if not line.startswith("    "):
            break
        fields = line.split()
        if fields[0] == wp.get_device(DEVICE).alias:
            return int(fields[1])
    return 0


_MEMORY_ACTIVITIES = {
    wp.TIMING_KERNEL_BUILTIN: "built-in kernel",
    wp.TIMING_MEMCPY: "memcpy",
    wp.TIMING_MEMSET: "memset",
    wp.TIMING_GRAPH: "graph",
}


def activities(function):
    """Device activities that ``function`` issues, read from Warp's activity timing (CUDA devices only).

    Returns the kernel launches by kernel name and the other activities by kind
    ("built-in kernel", "memcpy", "memset", "graph").
    """
    wp.synchronize()
    wp.timing_begin(cuda_filter=wp.TIMING_ALL)
    function()
    kernels, memory = collections.Counter(), collections.Counter()
    for result in wp.timing_end():
        if result.filter == wp.TIMING_KERNEL:
            # "forward kernel <key>"; a kernel made by a factory or in a module of its own carries a hash
            key = result.name.split()[-1].split("__locals__")[0]
            kernels[re.sub(r"_[0-9a-f]{8}$", "", key).strip("_")] += 1
        else:
            memory[_MEMORY_ACTIVITIES[result.filter]] += 1
    return kernels, memory


def scene_options(mjm):
    """The rigid options of the grasp scene: implicitfast and elliptic cones."""
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mjm.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC


# A position-controlled arm held against its joint limits, a box on the floor, a box on
# that box (a convex pair) and a ball: 21 degrees of freedom, 9 contacts, 3 limit rows.
ARM = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <default>
    <geom condim="3" friction="0.8 0.02 0.001"/>
    <joint damping="0.05" armature="0.01"/>
  </default>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-0.6 0.6" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
      <body pos="0.25 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-0.4 1.2" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025"/>
        <body pos="0.25 0 0">
          <joint name="wrist" type="slide" axis="0 0 1" range="-0.05 0.05" limited="true"/>
          <geom type="sphere" size="0.04"/>
        </body>
      </body>
    </body>
    <body pos="0.3 0.25 0.05"><freejoint/><geom type="box" size="0.08 0.08 0.05"/></body>
    <body pos="0.33 0.27 0.14"><freejoint/><geom type="box" size="0.04 0.05 0.04"/></body>
    <body pos="0.45 0 0.05"><freejoint/><geom type="sphere" size="0.05"/></body>
  </worldbody>
  <actuator>
    <position joint="shoulder" kp="40" kv="2"/>
    <position joint="elbow" kp="30" kv="1.5"/>
    <position joint="wrist" kp="200" kv="5"/>
  </actuator>
</mujoco>
"""

# A wrist on a hinge with two position-controlled jaws that hold a free capsule by friction,
# as a tool holds the needle, and a free box on the floor: 15 degrees of freedom.
GRIPPER = """
<mujoco>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" jacobian="dense"/>
  <default>
    <geom condim="4" friction="1.2 0.02 0.001"/>
    <joint damping="0.1" armature="0.005"/>
  </default>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.3">
      <joint name="wrist" type="hinge" axis="1 0 0" range="-25 25"/>
      <geom type="box" size="0.02 0.05 0.02" contype="0" conaffinity="0"/>
      <body pos="0 0.03 -0.05">
        <joint name="left" type="slide" axis="0 1 0" range="-0.02 0.01"/>
        <geom type="box" size="0.02 0.005 0.03"/>
      </body>
      <body pos="0 -0.03 -0.05">
        <joint name="right" type="slide" axis="0 1 0" range="-0.01 0.02"/>
        <geom type="box" size="0.02 0.005 0.03"/>
      </body>
    </body>
    <body pos="0 0 0.245">
      <freejoint/>
      <geom type="capsule" fromto="-0.04 0 0 0.04 0 0" size="0.01" mass="0.01"/>
    </body>
    <body pos="0.2 0 0.03"><freejoint/><geom type="box" size="0.03 0.03 0.03"/></body>
  </worldbody>
  <actuator>
    <position joint="wrist" kp="20" kv="1"/>
    <position joint="left" kp="300" kv="5"/>
    <position joint="right" kp="300" kv="5"/>
  </actuator>
</mujoco>
"""

# Two links without contacts: a muscle with an activation state on a spatial tendon, a limited
# fixed tendon with armature, stiffness and damping, and a camera and a light that follow the links.
CABLES = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <worldbody>
    <light mode="targetbodycom" target="forearm" pos="0 -1 2"/>
    <site name="anchor" pos="0.25 0 1.15"/>
    <body pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.05" range="-60 60"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <site name="upper" pos="0.2 0 0.03"/>
      <camera mode="trackcom" pos="0 -1 0.2"/>
      <body name="forearm" pos="0.3 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.05" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02"/>
        <site name="lower" pos="0.1 0 0.03"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed armature="0.02" stiffness="2" damping="0.1" limited="true" range="-0.3 0.3">
      <joint joint="shoulder" coef="1"/>
      <joint joint="elbow" coef="-0.5"/>
    </fixed>
    <spatial name="cable" armature="0.01">
      <site site="anchor"/>
      <site site="upper"/>
      <site site="lower"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle tendon="cable" lengthrange="0.05 0.6"/>
    <position joint="elbow" kp="5" kv="0.2"/>
  </actuator>
</mujoco>
"""

# Four hinges held against their upper limits by gravity, three with friction loss, the last two
# coupled by an equality. The chain stays above the plane.
CHAIN = """
    <body pos="0 0.6 0.5">
      <joint name="h0" type="hinge" axis="0 1 0" range="-30 -1" frictionloss="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body pos="0.2 0 0">
        <joint name="h1" type="hinge" axis="0 1 0" range="-30 -1" frictionloss="0.02"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body pos="0.2 0 0">
          <joint name="h2" type="hinge" axis="0 0 1" range="-25 25"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
          <body pos="0.2 0 0">
            <joint name="h3" type="hinge" axis="0 1 0" range="-40 -2" frictionloss="0.01"/>
            <geom type="sphere" size="0.04" pos="0.1 0 0"/>
          </body>
        </body>
      </body>
    </body>
"""

# A floor with the bodies and the chain that a scene puts on it.
FLOOR_SCENE = """
<mujoco>
  <option integrator="implicitfast" cone="elliptic" jacobian="dense"/>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    {bodies}
    {chain}
  </worldbody>
  <equality>
    <joint joint1="h2" joint2="h3" polycoef="0 0.5"/>
  </equality>
</mujoco>
"""

# nv 22. A box rests on up to four contacts and the lying capsule on two, so the contacts of a
# body share Hessian entries.
BOXES = FLOOR_SCENE.format(
    chain=CHAIN,
    bodies="""
    <body pos="0 0 0.08"><freejoint/><geom type="box" size="0.05 0.06 0.07" condim="3"/></body>
    <body pos="0.4 0 0.08"><freejoint/><geom type="box" size="0.06 0.05 0.07" condim="6"/></body>
    <body pos="0.8 0 0.05" euler="0 90 0"><freejoint/><geom type="capsule" size="0.04 0.1" condim="4"/></body>
""",
)

# nv 27. MuJoCo Warp's humanoid: one tree with joint limits that stands on several contacts.
HUMANOID = Path(mjw.__file__).resolve().parent / "test_data" / "humanoid" / "humanoid.xml"


def _arm():
    """The arm pushes against its joint limits; two boxes and a ball rest on the floor."""
    mjm = mujoco.MjModel.from_xml_string(ARM)
    mjd = mujoco.MjData(mjm)
    mjd.ctrl[:] = (0.9, 1.5, 0.08)  # targets past the joint ranges
    mujoco.mj_step(mjm, mjd, 60)
    return mjm, mjd


def _gripper():
    """The jaws hold the capsule and the wrist has turned it against its joint limit."""
    mjm = mujoco.MjModel.from_xml_string(GRIPPER)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[1:3] = (-0.0149, 0.0149)  # closed on the capsule
    mjd.ctrl[:] = (0.0, -0.02, 0.02)
    mujoco.mj_step(mjm, mjd, 300)
    mjd.ctrl[0] = 0.8  # past the joint range
    mujoco.mj_step(mjm, mjd, 100)
    return mjm, mjd


def _boxes():
    """Free bodies of 3, 4 and 6 rows per contact, and a hinge chain with limits, friction loss and an equality."""
    mjm = mujoco.MjModel.from_xml_string(BOXES)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_step(mjm, mjd, 40)
    return mjm, mjd


def _humanoid():
    mjm = mujoco.MjModel.from_xml_path(str(HUMANOID))
    scene_options(mjm)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mjd.ctrl[:] = np.random.default_rng(7).uniform(-0.1, 0.1, mjm.nu)
    mujoco.mj_step(mjm, mjd, 5)
    return mjm, mjd


def _cables():
    """The muscle has pulled the links close to the limit of the fixed tendon and is being released."""
    mjm = mujoco.MjModel.from_xml_string(CABLES)
    mjd = mujoco.MjData(mjm)
    mjd.ctrl[:] = (0.6, 0.4)
    mujoco.mj_step(mjm, mjd, 45)
    mjd.ctrl[0] = 0.3
    return mjm, mjd


# name: native model in the state that a rollout of whole steps starts from, constraint rows of a world
STEP_MODELS = {
    "arm": (_arm, 96),
    "gripper": (_gripper, 96),
    "boxes": (_boxes, 96),
    "humanoid": (_humanoid, 128),
    "cables": (_cables, 16),
}


def step_model(name, worlds=1, edit=None):
    """Device model and data of one of ``STEP_MODELS`` in a moving contact state; equal for every call.

    Args:
        name: Key of ``STEP_MODELS``.
        worlds: Number of worlds of the data.
        edit: Optional function that changes the device model before the data is made.
    """
    prepare, njmax = STEP_MODELS[name]
    mjm, mjd = prepare()
    mujoco.mj_forward(mjm, mjd)
    mjd.qacc_warmstart[:] = mjd.qacc
    m = mjw.put_model(mjm)
    if edit:
        edit(m)
    d = mjw.put_data(mjm, mjd, nworld=worlds, njmax=njmax)
    # a value that no step leaves where energies are not computed
    d.energy.fill_(wp.vec2(3.0, 4.0))
    return m, d
