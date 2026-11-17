# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""One MuJoCo Warp step of a small rigid scene, issued for replay from a CUDA graph.

``RigidStep.step`` leaves in ``Data`` what ``mujoco_warp.step`` leaves there. It calls
the stages of that step from MuJoCo Warp, in the order of ``step`` and ``forward`` in
``mujoco_warp/_src/forward.py`` (MuJoCo Warp 3.17.0, Apache-2.0), except for three
things that make a replayed step expensive or impossible to capture:

- The body-tree passes ``com_pos``, ``crb`` and ``rne`` take one launch each
  (``TreePasses``) instead of one launch per tree level.
- The constraint solve runs on a solver context that is kept between steps and
  fuses launches that always follow one another (``NewtonSolve``).
- The temporary arrays that the other stages create take the buffers of a pool
  (``StepScratch``). A captured step then holds no allocation nodes, which Warp
  requires of the body of a graph conditional.

Stages that do nothing in the supported configuration are not called: sleeping,
flex, sensors and the branches of the DISCRETE integrator. Three nodes that do no
work in a model like the needle and tool model remain, because they are inside
MuJoCo Warp functions that the step calls whole: the launch of ``_next_activation``
without activation states (``forward._advance``), the launch of
``_qderiv_tendon_damping`` without tendons (``derivative.deriv_smooth_vel``) and the
zeroing of ``qfrc_adhesion`` without adhesion actuators (``passive.passive``). Replayed
from a graph they take about 1 % of the time of a step of that model.
"""

import copy
import dataclasses

import warp as wp
from mujoco_warp._src import collision_driver, constraint, forward, passive, smooth
from mujoco_warp._src.types import BroadphaseType, DisableBit, EnableBit, IntegratorType

from ...fem.differentiation import active_tape
from .constraint_solver import NewtonSolve
from .scratch import StepScratch
from .tree import TreePasses

# Arrays of Data that hold the integration state and the inputs of a step.
_STATE = ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied", "time", "qacc", "qacc_warmstart", "qLD")
# The state arrays that a tape could differentiate a step with respect to.
_DIFFERENTIABLE = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
# What one step hands to the next and what a caller sets between steps: the integration state of
# MuJoCo (mjSTATE_INTEGRATION) and the overflow flags, which every step adds to.
_CARRIED = (
    "time",
    "qpos",
    "qvel",
    "act",
    "history",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "overflow",
)


def companion_data(data):
    """A second ``Data`` for the same model that shares what steps carry and owns every other array.

    The shared arrays are the integration state, the inputs and the overflow flags of
    ``data``. A step on either object therefore continues from the step before it on the
    other one, while the arrays it computes from that state (frames, inertia, contacts,
    constraint rows, forces) stay with the object it ran on. ``RigidStep.step`` can then
    skip the stages whose results one object still holds; see its ``state_stages`` argument.
    """

    def own(value):
        if isinstance(value, wp.array):
            return wp.clone(value)
        if dataclasses.is_dataclass(value):
            clone = copy.copy(value)
            for field in dataclasses.fields(value):
                setattr(clone, field.name, own(getattr(value, field.name)))
            return clone
        if isinstance(value, (tuple, list)):
            return type(value)(own(item) for item in value)
        return value

    companion = own(data)
    for name in _CARRIED:
        setattr(companion, name, getattr(data, name))
    return companion


def _check_supported(m, d):
    """Raise ``ValueError`` unless the stages that ``RigidStep.step`` calls make up the step of the model."""
    opt = m.opt
    required = (
        (opt.integrator == IntegratorType.IMPLICITFAST, "the implicitfast integrator"),
        (opt.broadphase == BroadphaseType.NXN, "the NXN broadphase"),
        (m.nsensor == 0, "no sensors"),
        (m.callback.sensor is None, "no sensor callback"),
        (m.nflex == 0, "no flex"),
        (not opt.enableflags & EnableBit.SLEEP, "sleeping disabled"),
        (opt.noslip_iterations == 0, "no noslip iterations"),
        (not any(getattr(d, name).requires_grad for name in _DIFFERENTIABLE), "state arrays without gradients"),
        (active_tape() is None, "no active tape"),
    )
    missing = [name for met, name in required if not met]
    if missing:
        raise ValueError("RigidStep requires " + ", ".join(missing))


def _configuration(m, d):
    """What a ``RigidStep`` is created for, by name.

    The sizes of the arrays it keeps and of the temporaries in its pool, the options of the
    supported configuration, and the memory of the state arrays.
    """
    opt = m.opt
    configuration = {
        "nworld": d.nworld,
        "njmax": d.njmax,
        "naconmax": d.naconmax,
        "naccdmax": d.naccdmax,
        "nv": m.nv,
        "nv_pad": m.nv_pad,
        "nC": m.nC,
        "nbody": m.nbody,
        "ngeom": m.ngeom,
        "nactuator": m.nactuator,
        "ntendon": m.ntendon,
        "nwrap": m.nwrap,
        "nflex": m.nflex,
        "nsensor": m.nsensor,
        "callback.sensor": m.callback.sensor is None,
        "is_sparse": m.is_sparse,
        "M.shape": d.M.shape,
        "qLD.shape": d.qLD.shape,
        "opt.integrator": opt.integrator,
        "opt.solver": opt.solver,
        "opt.cone": opt.cone,
        "opt.broadphase": opt.broadphase,
        "opt.noslip_iterations": opt.noslip_iterations,
        "opt.enableflags": opt.enableflags,
        "opt.disableflags": opt.disableflags,
        "opt.deterministic": opt.deterministic,
    }
    for name in _STATE:
        array = getattr(d, name)
        configuration[name] = (array.ptr, array.shape, array.dtype, array.requires_grad)
    return configuration


class RigidStep:
    """``mujoco_warp.step`` for one ``Model`` and ``Data``, with fused launches and kept temporaries.

    Supported, and checked when the object is created: the implicitfast integrator, the
    NXN broadphase (the sweep-and-prune broadphases allocate device memory that no Warp
    allocator serves), no sensors, sensor callback, flex, sleeping or noslip iterations,
    state arrays without gradients and no active tape, and what ``TreePasses`` and
    ``NewtonSolve`` require: the Newton solver with elliptic friction cones and a dense
    constraint Jacobian, at most 32 degrees of freedom, 256 constraint rows per world and
    256 bodies per tree level, and no deterministic atomics.

    Create it outside any graph capture and run ``step`` once outside capture before
    capturing it: that step compiles the kernels and sizes the pool of temporaries. A
    captured graph addresses the pool, the solver context and the tree tables by
    pointer, so keep the object alive while such a graph is launched and put
    ``signature`` into the key of a graph cache. Steps of one object must not run
    concurrently, for example in two branches of one graph.

    Args:
        model: MuJoCo Warp ``Model`` of the rigid scene.
        data: MuJoCo Warp ``Data`` of that model; ``step`` reads and writes its arrays.

    Attributes:
        tree: The ``TreePasses`` of the step.
        constraint_solve: The ``NewtonSolve`` of the step; ``constraint_solve.context`` is
            its solver context.
        state_scratch: The ``StepScratch`` that serves the temporary arrays of the state stages.
        scratch: The ``StepScratch`` that serves the temporary arrays of the rest of the step.
            The two parts have a pool each, because a pool serves one fixed sequence of
            requests and a step can leave the state stages out.

    Raises:
        ValueError: If the model or data is outside the supported configuration.
        RuntimeError: If the device of ``data`` is capturing a graph.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.device = data.qpos.device
        if self.device.is_capturing:
            raise RuntimeError("Create RigidStep outside graph capture")
        _check_supported(model, data)
        self.tree = TreePasses(model, data)
        self.constraint_solve = NewtonSolve(model, data)
        self.state_scratch = StepScratch(self.device)
        self.scratch = StepScratch(self.device)
        self._configuration = _configuration(model, data)
        self._warmed = False

    @property
    def signature(self):
        """Hashable identity of the buffers that a captured ``step`` addresses, for graph cache keys.

        Changes when a pool of temporaries adds or replaces a buffer. Two objects never
        have equal signatures.
        """
        return (self.state_scratch.signature, self.scratch.signature)

    def validate(self, model, data):
        """Raise ``RuntimeError`` unless ``model`` and ``data`` are what this object was created for.

        Compares the sizes, the options that the supported configuration names and the
        state arrays of ``data`` with those at creation, without reading device memory,
        and names what differs. ``step`` calls it; call it as well before other work that
        relies on the object, such as reading the solver context.
        """
        if model is not self.model or data is not self.data:
            raise RuntimeError("Another rigid model or data; create a new RigidStep outside graph capture")
        configuration = _configuration(model, data)
        changed = [name for name, value in self._configuration.items() if configuration[name] != value]
        if changed:
            raise RuntimeError(
                f"The rigid model, data or options changed ({', '.join(changed)}); "
                "create a new RigidStep outside graph capture"
            )
        if active_tape() is not None:
            raise RuntimeError("RigidStep does not support an active tape")

    def step(self, after_solver_loop=None, *, state_stages=True):
        """Advance ``data`` by one step, with the results of ``mujoco_warp.step(model, data)``.

        Every launch is on the device of ``data``, whatever the default device of the caller is.

        Args:
            after_solver_loop: Optional callable, called once right after the iteration loop
                of the constraint solve has been issued (``after_loop`` of
                ``NewtonSolve.solve``). A device array that it creates takes a buffer of the pool.
            state_stages: Run the stages that ``state_stages()`` runs. Pass ``False`` only
                if ``data`` still holds their results for its current position, velocity,
                activation and controls, as after a call of ``state_stages()``.

        Raises:
            RuntimeError: If the model, data or options are not those of creation
                (``validate``), or if the device is capturing a graph and no step has
                run outside capture.
        """
        m, d = self.model, self.data
        self._check_ready()
        with wp.ScopedDevice(self.device):
            if state_stages:
                with self.state_scratch.scope():
                    self._state_stages()
            with self.scratch.scope():
                # the rest of forward.forward and the integrator
                forward.fwd_acceleration(m, d, factorize=True)
                self.constraint_solve.solve(after_loop=after_solver_loop)
                if m.opt.run_rne_postconstraint:
                    smooth.rne_postconstraint(m, d)

                forward.implicit(m, d)
        self._warmed = True

    def state_stages(self):
        """Run the stages of a step that read only the position, velocity, activation and controls.

        These are ``fwd_position``, ``fwd_velocity`` and ``fwd_actuation`` of MuJoCo Warp:
        frames, inertia, contacts, constraint rows, bias, passive and actuator forces. They
        read none of the applied forces and not the warm start, so their results hold for
        every step that starts from the same state, whatever forces it applies. Such a step
        is ``step(state_stages=False)``.

        Raises:
            RuntimeError: As ``step``.
        """
        self._check_ready()
        with wp.ScopedDevice(self.device), self.state_scratch.scope():
            self._state_stages()

    def _check_ready(self):
        self.validate(self.model, self.data)
        if self.device.is_capturing and not self._warmed:
            raise RuntimeError("Run RigidStep.step once outside graph capture before capturing it")

    def _state_stages(self):
        """fwd_position, fwd_velocity and fwd_actuation in their order, on the current device and allocator.

        Left out are the stages of sleeping, flex and sensors, which launch nothing here, and the
        zeroing of the sensor data, which is an empty array and would still be a node of a captured graph.
        """
        m, d = self.model, self.data
        # forward.fwd_position without factorization and its fwd_kinematics, written out because
        # tree.com_pos and tree.crb stand for their smooth.com_pos and smooth.crb
        smooth.kinematics(m, d)
        self.tree.com_pos()
        smooth.camlight(m, d)
        smooth.tendon(m, d)
        self.tree.crb()
        smooth.tendon_armature(m, d)
        if m.opt.run_collision_detection:
            collision_driver.collision(m, d)
        constraint.make_constraint(m, d)
        smooth.transmission(m, d)
        forward._energy_pos(m, d)

        # forward.fwd_velocity, written out because tree.rne stands for its smooth.rne
        wp.launch(
            forward._actuator_velocity,
            dim=(d.nworld, m.nactuator),
            inputs=[d.qvel, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment],
            outputs=[d.actuator_velocity],
        )
        wp.launch(
            forward._tendon_velocity,
            dim=(d.nworld, m.ntendon),
            inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.qvel, d.ten_J],
            outputs=[d.ten_velocity],
        )
        smooth.com_vel(m, d)
        passive.passive(m, d)
        self.tree.rne()
        smooth.tendon_bias(m, d, d.qfrc_bias)
        forward._energy_vel(m, d)

        if m.callback.control and not m.opt.disableflags & DisableBit.ACTUATION:
            m.callback.control(m, d)
        forward.fwd_actuation(m, d)
