"""Isolated articulated MuJoCoWarp trial prototype; not a production backend.

Newton owns coordinate/control conversions. MuJoCoWarp owns rigid dynamics,
contacts, finite actuators and warm starts. Four microsteps hold the imported
world COM wrenches and scalar actuator targets, then export one endpoint.
Every microstep is one ``RigidStep.step``: the MuJoCo Warp step with fused
launches and persistent temporaries, which a conditional CUDA graph can hold.
"""

import dataclasses
import math
import threading
from typing import Any

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_solver_suturebot._src.solvers.coupled.interface.insertion import NeedleStepTooLarge
from newton_solver_suturebot._src.solvers.fem.differentiation import active_tape
from newton_solver_suturebot._src.utils.capture import ScopedCapture, note_release
from newton_solver_suturebot._src.utils.step_time import StepTime

# MuJoCo 3.17 mjSTATE_INTEGRATION, excluding unsupported plugin state.
# qacc is scratch: MuJoCoWarp resets it from warm-start/smooth acceleration.
INTEGRATION_FIELDS = (
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
)

_STATE_FIELDS = (
    "joint_q",
    "joint_qd",
    "body_q",
    "body_qd",
    "body_f",
    "particle_q",
    "particle_qd",
    "particle_f",
    "body_qdd",
    "body_parent_f",
)
_CONTROL_FIELDS = ("joint_f", "joint_target_q", "joint_target_qd", "tet_activations")
_BUDGET_FIELDS = (
    "iterations",
    "ls_iterations",
    "tolerance",
    "ls_tolerance",
    "integrator",
    "cone",
    "impratio",
    "solver",
)


class WarpRigidStepRejected(NeedleStepTooLarge):
    """GPU rigid integration left MuJoCo's valid range; the enclosing transaction must refine/retry."""


def _array_binding(array):
    return (
        id(array),
        array.ptr,
        array.shape,
        array.strides,
        array.dtype,
        array.device.alias,
    )


def _require_primal(model=None, arrays=()):
    if active_tape() is not None or getattr(model, "requires_grad", False):
        raise ValueError("Articulated MuJoCoWarp trials do not implement automatic derivatives")
    if any(array is not None and (array.requires_grad or array.grad is not None) for array in arrays):
        raise ValueError("Gradient-enabled trial arrays are unsupported by MuJoCoWarp")


class _DeviceArrayIndex:
    """The device arrays of MuJoCo Warp model and data, with a test that none was replaced.

    The preflight of every interval must know that the arrays a captured graph addresses
    are still the ones the model and data hold. Walking the nested dataclasses and
    describing 600 arrays costs more host time than the rest of the preflight, so the
    walk is done once and later calls only compare what each slot holds with what it held.
    A Warp array keeps its memory, shape and type for its lifetime, so an unchanged
    object is an unchanged binding. Only dataclass fields and tuples are followed, never
    array contents or other objects such as devices, BVHs or Python callbacks.
    """

    def __init__(self, *owners):
        self._slots = []  # (owner, field name or tuple index, value)
        arrays = []
        for owner in owners:
            self._index(owner, arrays)
        self.arrays = tuple(arrays)

    def _index(self, owner, arrays):
        if isinstance(owner, tuple):
            items = enumerate(owner)
        else:
            items = ((field.name, getattr(owner, field.name)) for field in dataclasses.fields(owner))
        for key, value in items:
            if isinstance(value, wp.array):
                arrays.append(value)
            elif dataclasses.is_dataclass(value) or isinstance(value, tuple):
                self._index(value, arrays)
            elif isinstance(value, list):
                raise TypeError("MuJoCo Warp model and data hold their arrays in dataclasses and tuples")
            else:
                continue
            if not isinstance(owner, tuple):
                # a tuple cannot change; the slot that holds the tuple is compared instead
                self._slots.append((owner, key, value))

    def unchanged(self):
        """Whether every dataclass field that held an array or a container still holds that object."""
        return all(getattr(owner, name) is value for owner, name, value in self._slots)


def _world_mask(selected, worlds, device):
    if selected is not None and (
        not isinstance(selected, wp.array)
        or selected.shape != (worlds,)
        or selected.dtype is not wp.bool
        or selected.device != device
        or not selected.is_contiguous
    ):
        raise ValueError("World mask must be a contiguous same-device Boolean array with one entry per world")


@wp.kernel(enable_backward=False)
def _grasp_contact_metrics(
    nacon: wp.array[int],
    contact_world: wp.array[int],
    contact_geom: wp.array[wp.vec2i],
    contact_address: wp.array2d[int],
    geom_body: wp.array[int],
    contact_force: wp.array[wp.spatial_vector],
    needle_body: int,
    jaw_bodies: wp.vec2i,
    metrics: wp.array2d[float],
):
    # Each jaw owns its reduction. No atomic ordering changes the diagnostic
    # sum, and no host contact-buffer copy or native forward is involved.
    world, jaw = wp.tid()
    normal = float(0.0)  # noqa: UP018 - Warp dynamic-loop accumulator
    count = int(0)  # noqa: UP018, RUF046 - Warp dynamic-loop accumulator
    # Rejected overflow trials can retain an attempted count above capacity.
    # Never read outside allocated contact scratch; acceptance still rejects
    # every overflow before these metrics can be committed by the coupler.
    if nacon[0] < 0 or nacon[0] > contact_geom.shape[0]:
        metrics[world, jaw] = wp.nan
        metrics[world, jaw + 2] = -1.0
        return
    for index in range(wp.min(nacon[0], contact_geom.shape[0])):
        if contact_world[index] == world:
            geoms = contact_geom[index]
            if geoms[0] >= 0 and geoms[1] >= 0:
                first = geom_body[geoms[0]]
                second = geom_body[geoms[1]]
                if (first == needle_body and second == jaw_bodies[jaw]) or (
                    second == needle_body and first == jaw_bodies[jaw]
                ):
                    force = contact_force[index][0]
                    normal += wp.max(0.0, force)
                    count += int(contact_address[index, 0] >= 0 and force > 0.0)
    metrics[world, jaw] = normal
    metrics[world, jaw + 2] = float(count)


@wp.kernel(enable_backward=False)
def _copy_selected_worlds(
    source: wp.array(dtype=Any),
    target: wp.array(dtype=Any),
    selected: wp.array[bool],
    items_per_world: int,
):
    index = wp.tid()
    if selected[index // items_per_world]:
        target[index] = source[index]


@wp.kernel(enable_backward=False)
def _set_microstep(microstep: wp.array[float], timestep: wp.array[float]):
    # MuJoCo Warp reads opt.timestep from this array in every kernel. The value comes
    # from device data, so a captured interval is not tied to the time step of its capture.
    timestep[wp.tid()] = microstep[0]


@wp.kernel(enable_backward=False)
def _load_trial_warm_start(valid: wp.array[int], saved: wp.array2d[float], warm_start: wp.array2d[float]):
    world, dof = wp.tid()
    if valid[0] != 0:
        warm_start[world, dof] = saved[world, dof]


@wp.kernel(enable_backward=False)
def _exchange_trial_warm_start(
    valid: wp.array[int],
    warm_start: wp.array2d[float],
    finished: wp.array2d[float],
    following: wp.array2d[float],
):
    # After one microstep the backend's warm start is that microstep's solution.
    # Keep it for the next trial and start the next microstep from the solution
    # the previous trial found for it.
    world, dof = wp.tid()
    finished[world, dof] = warm_start[world, dof]
    if valid[0] != 0:
        warm_start[world, dof] = following[world, dof]


@wp.kernel(enable_backward=False)
def _store_trial_warm_start(warm_start: wp.array2d[float], finished: wp.array2d[float], valid: wp.array[int]):
    world, dof = wp.tid()
    finished[world, dof] = warm_start[world, dof]
    valid[0] = 1


class WarpIntegrationCheckpoint:
    """Device-only transactional integration-state copies; no host readback.

    Sleeping is deliberately excluded: it additionally retains cached kinematics,
        awake lists and counters. This prototype has no stateful user callbacks or
        plugins. Contacts/Jacobians are forward scratch, regenerated every trial.
    """

    def __init__(self, solver):
        if solver.use_mujoco_cpu or solver.enable_sleeping or solver.mj_model.nplugin:
            raise ValueError("Checkpoint requires non-sleeping, plugin-free MuJoCoWarp")
        self.solver = solver
        self.data = solver.mjw_data
        self.worlds = self.data.nworld
        if type(self.worlds) is not int or self.worlds < 1:
            raise ValueError("Checkpoint needs a positive integer world count")
        self.arrays = tuple(getattr(self.data, name) for name in INTEGRATION_FIELDS)
        _require_primal(getattr(solver, "model", None), self.arrays)
        for name, array in zip(INTEGRATION_FIELDS, self.arrays, strict=True):
            if not isinstance(array, wp.array) or not array.is_contiguous or array.shape[0] != self.worlds:
                raise ValueError(f"Checkpoint field {name} must have contiguous world-leading storage")
            if array.device != self.arrays[0].device:
                raise ValueError("Checkpoint fields must share a device")
        self.bindings = tuple(_array_binding(array) for array in self.arrays)
        self.saved = tuple(wp.empty_like(array, requires_grad=False) for array in self.arrays)
        self.initialized = False

    def _validate(self):
        _require_primal(getattr(self.solver, "model", None), self.arrays)
        if (
            self.solver.mjw_data is not self.data
            or self.data.nworld != self.worlds
            or any(
                getattr(self.data, name) is not array
                for name, array in zip(INTEGRATION_FIELDS, self.arrays, strict=True)
            )
            or tuple(_array_binding(array) for array in self.arrays) != self.bindings
        ):
            raise ValueError("Rebuild checkpoint after MuJoCoWarp data storage replacement")

    def _copy(self, targets, sources, selected):
        _world_mask(selected, self.worlds, self.arrays[0].device)
        for target, source in zip(targets, sources, strict=True):
            if not source.size:
                continue
            if selected is None:
                wp.copy(target, source)
            else:
                wp.launch(
                    _copy_selected_worlds,
                    source.size,
                    [
                        source.flatten(),
                        target.flatten(),
                        selected,
                        source.size // self.worlds,
                    ],
                    device=source.device,
                )

    def save(self, selected=None):
        self._validate()
        if selected is not None and not self.initialized:
            raise ValueError("Save every world once before using a masked checkpoint")
        self._copy(self.saved, self.arrays, selected)
        self.initialized = True
        # A new accepted start: the previous interval's trial solutions no longer apply.
        reset = getattr(self.solver, "reset_trial_warm_start", None)
        if reset is not None:
            reset()

    def restore(self, selected=None):
        self._validate()
        if not self.initialized:
            raise ValueError("Save the checkpoint before restoring it")
        # A prepared observation describes the discarded trial, not this checkpoint.
        self.solver._prepared_observations = None
        self._copy(self.arrays, self.saved, selected)


@dataclasses.dataclass(frozen=True)
class PreparedObservations:
    """Single-use accepted-state snapshot; never an integration checkpoint."""

    metrics: object
    actual_clock: object
    expected_clock: object
    generation: int
    dt: float | None


class SubcycledMuJoCoWarp(SolverMuJoCo):
    """Narrow GPU counterpart of the existing native batched rigid adapter.

    No native get/setState or CPU conversion occurs in step(). The fixed Python
        microstep loop is CUDA-capturable. `_step` is not advanced: this prototype
        requires synchronization on every interval and has no step-count callbacks.
        Accepted clocks and counters advance only in commit_observations().
    """

    def __init__(self, model, *, rigid_substeps=4, gpu_tolerance=1e-6, trial_warm_start=True, **kwargs):
        if type(rigid_substeps) is not int or rigid_substeps < 1:
            raise ValueError("rigid_substeps must be a positive integer")
        if (
            isinstance(gpu_tolerance, (bool, np.bool_))
            or not isinstance(gpu_tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(gpu_tolerance)
            or gpu_tolerance <= 0
        ):
            raise ValueError("gpu_tolerance must be finite and positive")
        self.gpu_tolerance = float(gpu_tolerance)
        if kwargs.get("use_mujoco_cpu", False):
            raise ValueError("This adapter uses MuJoCoWarp")
        if not model.device.is_cuda or model.world_count < 1:
            raise ValueError("Prototype requires one or more CUDA worlds")
        _require_primal(model)
        self._capture_ready = False
        kwargs["use_mujoco_cpu"] = False
        super().__init__(model, **kwargs)
        # Explicit FP32-backend stopping contract. The native requested option
        # remains separately reported; matching it is not an accuracy claim.
        # 1e-6 is the documented MuJoCo Warp precision-aware default.
        self.mjw_model.opt.tolerance.fill_(self.gpu_tolerance)
        self.effective_gpu_options = self.audit_gpu_options()
        self.captured_gpu_options = None
        if self.enable_sleeping or self._kinematic_weld_eq is not None or self._use_speculative_contacts:
            raise ValueError("Prototype excludes sleeping, kinematic welds and speculative contacts")
        if not self.mjw_model.opt.run_collision_detection:
            raise ValueError("Prototype uses MuJoCoWarp collision detection")
        if self.mj_model.nplugin or self.mj_model.nflex:
            raise ValueError("Prototype excludes plugins and MuJoCo flex dynamics")
        if any(
            getattr(self.mjw_model.callback, name) is not None
            for name in (
                "passive",
                "control",
                "act_dyn",
                "act_gain",
                "act_bias",
                "sensor",
                "contactfilter",
            )
        ):
            raise ValueError("Prototype excludes stateful custom MuJoCoWarp callbacks")
        allowed = [
            newton.JointType.FIXED,
            newton.JointType.PRISMATIC,
            newton.JointType.REVOLUTE,
            newton.JointType.FREE,
        ]
        if not np.isin(model.joint_type.numpy(), allowed).all():
            raise ValueError("Held controls require scalar joints and free joints")
        self.rigid_substeps = rigid_substeps
        # The interval dt as device data. Its fraction dt / rigid_substeps is what every
        # issued interval copies into MuJoCo Warp's opt.timestep (_set_microstep).
        self.step_time = StepTime(model.device)
        self.step_time.fraction(rigid_substeps)
        # Repeated trials of one interval solve nearly the same constraint problem
        # in each microstep. Starting each microstep's constraint solve from the
        # solution the previous trial found for it changes only the solver's
        # initial iterate; tolerance, iteration limits and every restored
        # integration quantity are those of an unassisted trial.
        self.trial_warm_start = bool(trial_warm_start)
        self._trial_warm_slots = None
        self._trial_warm_valid = None
        self.native_batches = self.general_intervals = 0
        self._native_transfer = None
        self._rigid_graphs = {}
        self._rigid_warmed = False
        self.gpu_graph_captures = self.gpu_graph_launches = 0
        self.gpu_guard_rejections = 0
        self._shadow_token = None
        self._contact_metric_buffers = None
        self._grasp_body_ids = None
        self.accepted_grasp_metrics = None
        self._prepared_grasp_metrics = None
        self._capture_generation = 0
        self._capture_storage = None
        self._capture_index = None
        self._capture_owners = ()
        # The step of every microstep; created by the first warm-up, outside capture.
        self.rigid_step = None
        # The step of the first microstep of an interval, on a second Data that shares the
        # integration state. Trials of one interval start from one state, so the stages of
        # their first microstep that read only that state are issued once (prepare_trials).
        self._first_step = None
        self._prepared_overflow = None
        self._execution_thread = None
        self._execution_stream = None
        self._accepted_clock = np.zeros(self.mjw_data.nworld, dtype=np.float32)
        self._actual_accepted_clock = self._accepted_clock.copy()
        self._accepted_intervals = np.zeros(self.mjw_data.nworld, dtype=np.int64)
        self._accepted_generation = 0
        self._prepared_observations = None
        self._trial_dt = None
        self._capture_ready = True

    def _validate_execution_context(self):
        """Keep shared integration state and the rigid step on one owner and stream.

        Host preflight binds the first execution thread and ordered stream.
        Conditional capture can temporarily use an internal capture stream;
        it does not launch work, so only its owner thread is checked here.
        Every later host preflight/step checks the actual replay stream again.
        This is not a lock: concurrent trials sharing this Data and its rigid
        step are unsupported. While a microstep is issued, the pool of the rigid
        step serves every array created on the device by this thread.
        """
        device = self.model.device
        thread = threading.current_thread()
        owner = self._execution_thread
        if owner is not None and thread is not owner:
            raise RuntimeError("Rigid trials must use their owning Python thread")
        if device.is_capturing:
            if owner is None:
                raise RuntimeError("Preflight rigid execution outside CUDA capture")
            return
        # CPU-only contract fixtures have no CUDA stream; construction of a
        # real adapter already requires a CUDA model.
        stream = device.stream if device.is_cuda else None
        if owner is None:
            self._execution_thread = thread
            self._execution_stream = stream
        elif stream is not self._execution_stream and (
            stream is None or self._execution_stream is None or stream.cuda_stream != self._execution_stream.cuda_stream
        ):
            raise RuntimeError("Rigid trials must use their original ordered CUDA stream")

    def adopt_step_time(self, step_time):
        """Read the interval from ``step_time``, the ``StepTime`` of an enclosing coupled step.

        All stages of that step then read one set of arrays, which its owner writes
        once. The graphs of this solver address the arrays they were captured with
        and are dropped when another object takes their place.
        """
        if step_time is self.step_time:
            return
        if self.model.device.is_capturing:
            raise RuntimeError("Share the step time outside CUDA graph capture")
        step_time.fraction(self.rigid_substeps)
        self.step_time = step_time
        note_release(len(self._rigid_graphs))
        self._rigid_graphs.clear()

    def _write_microstep(self, dt):
        """Issue the copy of ``dt / rigid_substeps`` from the step time into ``opt.timestep``.

        ``StepTime.set`` writes a changed interval on the host, outside capture; inside a
        capture it only verifies that the host preparation wrote this interval. The
        copy itself is a node of the capture, so every replay restores the microstep
        whatever was written to ``opt.timestep`` in between.
        """
        self.step_time.set(dt)
        timestep = self.mjw_model.opt.timestep
        wp.launch(
            _set_microstep,
            timestep.shape[0],
            [self.step_time.fraction(self.rigid_substeps), timestep],
            device=self.model.device,
        )

    def invalidate_rigid_graphs(self):
        """Invalidate after external model/options edits, before the next trial.

        In-place writes to device option arrays cannot be detected from their
        Python identity. Call this method (or notify_model_changed) after such
        edits. The next host preflight audits the effective physical budget.
        No checkpoint or accepted state is modified by invalidation.
        """
        self._validate_execution_context()
        if self.model.device.is_capturing:
            raise ValueError("Invalidate rigid graphs outside CUDA capture")
        note_release(len(self._rigid_graphs))
        self._rigid_graphs.clear()
        # A graph of an enclosing capture may still address the buffers of the
        # current rigid step. Such a graph keeps the step alive; the next
        # warm-up creates a new one.
        self.rigid_step = None
        self._first_step = None
        self._prepared_overflow = None
        self._rigid_warmed = False
        self._capture_storage = None
        self._capture_owners = ()
        self._contact_metric_buffers = None
        self._shadow_token = None
        self._trial_warm_slots = None
        self._trial_warm_valid = None
        self._capture_generation += 1

    def reset_trial_warm_start(self):
        """Forget saved trial solutions; the next trial starts as an unassisted step."""
        valid = getattr(self, "_trial_warm_valid", None)
        if valid is not None:
            if self.model.device.is_capturing:
                raise RuntimeError("Reset trial warm starts outside CUDA capture")
            valid.zero_()

    def notify_model_changed(self, flags):
        ready = getattr(self, "_capture_ready", False)
        if ready:
            self._validate_execution_context()
        if ready and self.model.device.is_capturing:
            raise ValueError("Notify articulated model changes outside CUDA capture")
        result = super().notify_model_changed(flags)
        if ready:
            self.invalidate_rigid_graphs()
        return result

    def _validate_trial_arguments(self, state_in, state_out, control, dt):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0 or state_in is state_out:
            raise ValueError("Distinct states and a finite positive interval are required")
        if control is None:
            raise ValueError("GPU rigid trials require explicitly allocated controls")
        arrays = tuple(getattr(state, name, None) for state in (state_in, state_out) for name in _STATE_FIELDS)
        arrays += tuple(getattr(control, name, None) for name in _CONTROL_FIELDS)
        arrays += (
            getattr(getattr(control, "mujoco", None), "ctrl", None),
            getattr(getattr(state_out, "mujoco", None), "qfrc_actuator", None),
        )
        _require_primal(self.model, arrays)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            source, target = (
                getattr(state_in, name, None),
                getattr(state_out, name, None),
            )
            if source is not None and target is not None and source.size and source.ptr == target.ptr:
                raise ValueError(f"Rigid output {name} aliases the incoming state")
        return tuple(array for array in arrays if array is not None)

    def validate_capture_bindings(self, state_in, state_out, control, dt):
        """Host-only preflight and stable fingerprint for an outer trial graph.

        Unchanged bindings do not read device data. Scalar option/storage edits
        invalidate cached rigid graphs and re-audit budgets. External in-place
        device-option writes require invalidate_rigid_graphs first. Owners are
        retained while a graph references them; callers must invalidate their
        outer graph whenever this fingerprint changes. The interval ``dt`` is
        not part of the fingerprint: this call writes it to the step time, which
        a captured interval reads, so a graph serves every interval.
        """
        if self.model.device.is_capturing:
            raise ValueError("Validate rigid capture bindings before beginning capture")
        self._validate_execution_context()
        trial_arrays = self._validate_trial_arguments(state_in, state_out, control, dt)
        if self.enable_sleeping or self._kinematic_weld_eq is not None or self._use_speculative_contacts:
            raise ValueError("GPU trial contract excludes sleeping, kinematic welds and speculative contacts")
        if self.mj_model.nplugin or self.mj_model.nflex or not self.mjw_model.opt.run_collision_detection:
            raise ValueError("GPU trial contract requires rigid native contacts without plugins or flex")
        if any(
            getattr(self.mjw_model.callback, name) is not None
            for name in (
                "passive",
                "control",
                "act_dyn",
                "act_gain",
                "act_bias",
                "sensor",
                "contactfilter",
            )
        ):
            raise ValueError("GPU trial contract excludes custom callbacks")
        scalar_options = tuple(
            (field.name, value)
            for field in dataclasses.fields(self.mjw_model.opt)
            if isinstance(
                value := getattr(self.mjw_model.opt, field.name),
                (int, float, bool, str, type(None)),
            )
        )
        native_budget = tuple(getattr(self.mj_model.opt, name) for name in _BUDGET_FIELDS)
        settings = (
            id(self.mjw_model),
            id(self.mjw_data),
            self.rigid_substeps,
            float(self.gpu_tolerance),
            bool(getattr(self, "trial_warm_start", False)),
            native_budget,
            scalar_options,
        )
        index = self._capture_index
        if self._capture_storage is None or settings != self._capture_storage[:-1] or not index.unchanged():
            index = _DeviceArrayIndex(self.mjw_model, self.mjw_data)
            storage = (*settings, tuple(_array_binding(array) for array in index.arrays))
            if storage != self._capture_storage:
                self.invalidate_rigid_graphs()
                # Mismatched requested/effective budgets must fail before any trial.
                self.effective_gpu_options = self.audit_gpu_options()
                self._capture_storage = storage
                self._capture_owners = index.arrays
                if self._grasp_body_ids is not None:
                    self.grasp_contact_metrics(self._grasp_body_ids[0], self._grasp_body_ids[1:])
            self._capture_index = index
        _require_primal(self.model, index.arrays)
        storage = self._capture_storage
        self._trial_dt = float(dt)
        # Host preparation, outside capture: the captured interval reads the interval
        # from these arrays and checks, when it is captured, that they hold this dt.
        self.step_time.fraction(self.rigid_substeps)
        self.step_time.set(dt)
        self._prepared_observations = None
        steps = (self.rigid_step, self._first_step)
        for rigid_step in steps:
            if rigid_step is not None:
                rigid_step.validate(self.mjw_model, rigid_step.data)
        return (
            self._capture_generation,
            (
                storage,
                tuple(None if rigid_step is None else rigid_step.signature for rigid_step in steps),
            ),
            tuple(_array_binding(array) for array in trial_arrays),
        )

    def audit_gpu_options(self):
        """Read effective device budgets, including any backend default clamp."""
        result = {}
        effective = {}
        for name in (
            "iterations",
            "ls_iterations",
            "tolerance",
            "ls_tolerance",
            "integrator",
            "cone",
            "solver",
            "impratio_invsqrt",
        ):
            value = getattr(self.mjw_model.opt, name)
            effective[name] = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
            result[name] = effective[name].tolist()
        tolerance = np.asarray(result["tolerance"], dtype=np.float32)
        np.testing.assert_array_equal(
            tolerance,
            np.full(tolerance.shape, self.gpu_tolerance, dtype=np.float32),
            err_msg="Effective GPU tolerance must match the explicit GPU precision contract",
        )
        for name in (
            "iterations",
            "ls_iterations",
            "ls_tolerance",
            "integrator",
            "cone",
            "solver",
        ):
            actual = effective[name]
            wanted = np.asarray(getattr(self.mj_model.opt, name), dtype=actual.dtype)
            np.testing.assert_array_equal(actual, np.broadcast_to(wanted, actual.shape), err_msg=f"GPU opt.{name}")
        actual = np.asarray(result["impratio_invsqrt"], dtype=np.float32)
        wanted = np.full(actual.shape, 1.0 / np.sqrt(self.mj_model.opt.impratio), dtype=np.float32)
        np.testing.assert_array_equal(actual, wanted, err_msg="GPU opt.impratio")
        return result

    def initialize_from_newton(self, state, control):
        """Reset-only import; seed the private native shadow used for CPU IK."""
        self._apply_mjc_control(self.model, state, control, self.mjw_data)
        self._update_mjc_data(self.mjw_data, self.model, state)
        self._accepted_clock = self.mjw_data.time.numpy().copy()
        self._actual_accepted_clock = self._accepted_clock.copy()
        self._accepted_intervals = np.zeros(self.mjw_data.nworld, dtype=np.int64)
        self._accepted_generation += 1
        self._prepared_observations = None
        self._trial_dt = None
        self.sync_native_shadow(forward=True)

    def sync_native_shadow(self, *, forward=False, token=None):
        """Refresh a CPU observer/approximate metric, never GPU integration.

        The shadow is not the physical rigid solver. Its endpoint contact
        forces are a separate native diagnostic, not recorded GPU forces.
        ``token`` permits one accepted-state refresh per control frame.
        """
        if token is not None and token == self._shadow_token:
            return
        if self.mjw_data.nworld != 1:
            raise ValueError("Native observer shadow currently supports one GPU world")
        self.mj_data.time = float(self.mjw_data.time.numpy()[0])
        for name in INTEGRATION_FIELDS[1:]:
            values = getattr(self.mjw_data, name)
            if values.size:
                getattr(self.mj_data, name)[:] = values.numpy()[0]
        if forward:
            self._mujoco.mj_forward(self.mj_model, self.mj_data)
        self._shadow_token = token

    def check_overflow(self):
        """An accepted rigid interval must not drop any physical constraints."""
        overflow = self.mjw_data.overflow.numpy()
        if np.any(overflow):
            from mujoco_warp._src.types import OverflowType

            bits = int(np.bitwise_or.reduce(np.asarray(overflow, dtype=np.int64).reshape(-1)))
            names = [flag.name for flag in OverflowType if bits & int(flag)]
            unknown = bits & ~int(OverflowType.ALL)
            if unknown:
                names.append(f"UNKNOWN_{unknown}")
            raise RuntimeError(
                "MuJoCoWarp rejected rigid trial (capacity or solver budget): "
                f"{', '.join(names)}; raw={overflow.tolist()}"
            )

    def reject_unstable_trial(self):
        """Reject a trial whose rigid coordinates left MuJoCo's valid range.

        Native MuJoCo warns and resets when a position, velocity or acceleration
        is not finite or exceeds ``mjMAXVAL``; the native adapter turns that into
        a rejected trial. MuJoCo Warp does not check, so a trial loop that sees a
        non-finite interface state asks here. One readback, on that path only.
        """
        limit = float(self._mujoco.mjMAXVAL)
        for name in ("qpos", "qvel", "qacc"):
            values = getattr(self.mjw_data, name).numpy()
            if not np.isfinite(values).all() or np.abs(values).max(initial=0.0) > limit:
                self.gpu_guard_rejections += 1
                raise WarpRigidStepRejected(f"MuJoCo Warp {name} left the valid range; reject the rigid trial")

    def grasp_contact_metrics(self, needle_body, jaw_bodies):
        """Actual last-microstep forces, reduced to four values per world.

        The returned device buffer is scratch. A cohort must copy a world's
        values at acceptance before further trials replace global contacts.
        """
        import mujoco_warp

        if self._contact_metric_buffers is None:
            count = self.mjw_data.contact.geom.size
            self._contact_metric_buffers = (
                wp.array(
                    np.arange(count, dtype=np.int32),
                    dtype=wp.int32,
                    device=self.model.device,
                ),
                wp.zeros(count, dtype=wp.spatial_vector, device=self.model.device),
                wp.empty(
                    (self.mjw_data.nworld, 4),
                    dtype=wp.float32,
                    device=self.model.device,
                ),
                wp.array(self.mj_model.geom_bodyid, dtype=wp.int32, device=self.model.device),
            )
        contact_ids, forces, metrics, geom_body = self._contact_metric_buffers
        mujoco_warp.contact_force(self.mjw_model, self.mjw_data, contact_ids, False, forces)
        wp.launch(
            _grasp_contact_metrics,
            (self.mjw_data.nworld, 2),
            [
                self.mjw_data.nacon,
                self.mjw_data.contact.worldid,
                self.mjw_data.contact.geom,
                self.mjw_data.contact.efc_address,
                geom_body,
                forces,
                needle_body,
                wp.vec2i(*jaw_bodies),
                metrics,
            ],
            device=self.model.device,
        )
        return metrics

    def bind_grasp_observer(self, needle_label, jaw_labels):
        """Configure the one grasp pair shared by all replicated worlds."""
        ids = tuple(
            self._mujoco.mj_name2id(self.mj_model, self._mujoco.mjtObj.mjOBJ_BODY, label.replace("/", "_"))
            for label in (needle_label, *jaw_labels)
        )
        if len(ids) != 3 or min(ids) < 0:
            raise ValueError("GPU grasp observer body labels must resolve to one needle and two jaws")
        self._grasp_body_ids = ids
        self.accepted_grasp_metrics = wp.zeros((self.mjw_data.nworld, 4), device=self.model.device)
        self._prepared_grasp_metrics = wp.empty_like(self.accepted_grasp_metrics)
        # Prepare public contact-force scratch before any transaction capture.
        self.grasp_contact_metrics(ids[0], ids[1:])

    def prepare_observations(self):
        """Validate actual clock/contact data before commit; rejection changes no reference."""
        self._validate_execution_context()
        self._prepared_observations = None
        self.check_overflow()
        actual = self.mjw_data.time.numpy().copy()
        reference = getattr(self, "_accepted_clock", None)
        dt = getattr(self, "_trial_dt", None)
        expected = None
        if reference is not None:
            if dt is None or reference.shape != actual.shape:
                raise RuntimeError("A prepared rigid interval requires an initialized clock and trial dt")
            expected = reference.copy()
            delta = np.float32(dt / self.rigid_substeps)
            for _ in range(self.rigid_substeps):
                expected = np.asarray(expected + delta, dtype=np.float32)
            if not np.array_equal(actual, expected):
                raise RuntimeError("GPU rigid clock differs from the independently accumulated accepted interval")
        if not np.isfinite(actual).all():
            raise FloatingPointError("Nonfinite GPU rigid clock")
        metrics = None
        if self._grasp_body_ids is not None:
            ids = self._grasp_body_ids
            scratch = self.grasp_contact_metrics(ids[0], ids[1:])
            if not np.isfinite(scratch.numpy()).all():
                raise FloatingPointError("Nonfinite GPU grasp contact metrics")
            wp.copy(self._prepared_grasp_metrics, scratch)
            metrics = self._prepared_grasp_metrics
        prepared = PreparedObservations(metrics, actual, expected, getattr(self, "_accepted_generation", 0), dt)
        self._prepared_observations = prepared
        return prepared

    def commit_observations(self, selected=None, *, prepared=None):
        """Publish one validated endpoint and its independent FP32 clock reference."""
        self._validate_execution_context()
        _world_mask(selected, self.mjw_data.nworld, self.model.device)
        prepared = self.prepare_observations() if prepared is None else prepared
        if prepared is not getattr(self, "_prepared_observations", None) or prepared.generation != getattr(
            self, "_accepted_generation", 0
        ):
            raise ValueError("Commit the current single-use snapshot returned by prepare_observations")
        mask = np.ones(self.mjw_data.nworld, dtype=bool) if selected is None else selected.numpy()
        metrics = prepared.metrics
        if metrics is not None:
            if selected is None:
                wp.copy(self.accepted_grasp_metrics, metrics)
            else:
                wp.launch(
                    _copy_selected_worlds,
                    metrics.size,
                    [
                        metrics.flatten(),
                        self.accepted_grasp_metrics.flatten(),
                        selected,
                        4,
                    ],
                    device=self.model.device,
                )
        if prepared.expected_clock is not None:
            self._accepted_clock[mask] = prepared.expected_clock[mask]
            self._actual_accepted_clock[mask] = prepared.actual_clock[mask]
            self._accepted_intervals[mask] += 1
        self._accepted_generation = prepared.generation + 1
        self._prepared_observations = None

    def diagnostics(self):
        """Last committed GPU evidence; no native shadow or fresh device readback."""
        reference = getattr(self, "_accepted_clock", None)
        actual = getattr(self, "_actual_accepted_clock", None)
        rigid_step = self.rigid_step
        return {
            "rigid_backend": "mujoco_warp",
            "rigid_substeps": self.rigid_substeps,
            "gpu_effective_options": self.effective_gpu_options,
            "native_requested_tolerance": float(self.mj_model.opt.tolerance),
            "gpu_requested_tolerance": self.gpu_tolerance,
            "trial_warm_start": bool(getattr(self, "trial_warm_start", False)),
            "gpu_accepted_clock_s": None if actual is None else actual.tolist(),
            "gpu_accepted_clock_reference_s": None if reference is None else reference.tolist(),
            "gpu_accepted_clock_matches": bool(reference is not None and np.array_equal(actual, reference)),
            "gpu_accepted_intervals": getattr(self, "_accepted_intervals", np.zeros(0, dtype=int)).tolist(),
            "gpu_scratch_buffers": 0
            if rigid_step is None
            else rigid_step.state_scratch.buffer_count + rigid_step.scratch.buffer_count,
            "gpu_graph_captures": self.gpu_graph_captures,
            "gpu_graph_launches": self.gpu_graph_launches,
            "gpu_guard_rejections": getattr(self, "gpu_guard_rejections", 0),
        }

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        """Capture each stable rigid input/output binding after one real warmup."""
        self._validate_execution_context()
        self._validate_trial_arguments(state_in, state_out, control, dt)
        if contacts is not None or observables is not None:
            raise ValueError("Prototype exports rigid state only; inspect mjw_data outside timed trials")
        if self.model.device.is_capturing:
            return self._step_uncaptured(state_in, state_out, control, contacts, dt, observables=observables)
        fingerprint = self.validate_capture_bindings(state_in, state_out, control, dt)
        self.mj_model.opt.timestep = dt / self.rigid_substeps
        if not self._rigid_warmed:
            self._step_uncaptured(state_in, state_out, control, contacts, dt, observables=observables)
            self._rigid_warmed = True
            return
        owners = tuple(
            getattr(state, name, None)
            for state, names in (
                (state_in, ("joint_q", "joint_qd", "body_f")),
                (
                    state_out,
                    (
                        "joint_q",
                        "joint_qd",
                        "body_q",
                        "body_qd",
                        "body_qdd",
                        "body_parent_f",
                    ),
                ),
                (control, ("joint_target_q", "joint_target_qd", "joint_f")),
            )
            for name in names
        ) + (
            getattr(getattr(control, "mujoco", None), "ctrl", None),
            getattr(getattr(state_out, "mujoco", None), "qfrc_actuator", None),
        )
        key = fingerprint
        cached = self._rigid_graphs.get(key)
        if cached is None:
            self.captured_gpu_options = self.audit_gpu_options()
            with ScopedCapture(device=self.model.device) as capture:
                self._step_uncaptured(state_in, state_out, control, contacts, dt, observables=observables)
            cached = (
                capture.graph,
                owners
                + self._capture_owners
                + (self.rigid_step, self._first_step, self._trial_warm_slots, self._trial_warm_valid),
            )
            self._rigid_graphs[key] = cached
            if len(self._rigid_graphs) > 32:
                self._rigid_graphs.pop(next(iter(self._rigid_graphs)))
                note_release()
            self.gpu_graph_captures += 1
        wp.capture_launch(cached[0])
        self.gpu_graph_launches += 1

    def capture_graph(self, state_in, state_out, control, dt, *, before_interval=None):
        """Leave a captured and launched graph for this binding of :meth:`step`; for start-up.

        :meth:`step` runs the first interval of a solver as written and captures the next
        interval of a binding. This issues those intervals ahead of the trials: at most three,
        each a real interval from ``state_in`` into ``state_out``. The first creates the rigid
        steps, sizes their pools and loads their kernel modules; the second is captured and its
        graph launched, which is where Warp builds the executable; a third is needed when the
        first interval switched a scalar option of the backend, which drops what the first
        prepared. A binding that has a graph is left alone, so a second binding of a warmed
        solver takes one interval.

        ``before_interval`` is called before each interval; the caller passes what returns the
        integration state to the start that every interval of its transaction shares, and
        calls it once more afterwards. The intervals write ``state_out`` and the scratch of
        the backend like any trial. ``gpu_graph_captures`` counts the capture;
        ``gpu_graph_launches`` and the timestep of the native model keep their values. The
        device copy of the time step (``mjw_model.opt.timestep``) is left at the microstep of
        ``dt``: every interval writes it before it reads it, and the start-up call of the
        coupler (``capture_graphs``) gives back the values it had.
        """
        launches, native_timestep = self.gpu_graph_launches, self.mj_model.opt.timestep
        try:
            for _ in range(3):
                if self.validate_capture_bindings(state_in, state_out, control, dt) in self._rigid_graphs:
                    return
                if before_interval is not None:
                    before_interval()
                self.step(state_in, state_out, control, None, dt)
            if self.validate_capture_bindings(state_in, state_out, control, dt) not in self._rigid_graphs:
                raise RuntimeError("The rigid interval of this binding has no graph after three intervals")
        finally:
            self.gpu_graph_launches = launches
            self.mj_model.opt.timestep = native_timestep

    def _step_uncaptured(
        self,
        state_in,
        state_out,
        control,
        contacts,
        dt,
        *,
        observables=None,
        after_solver_loop=None,
        first_state_prepared=False,
    ):
        """Issue the microsteps of one interval, eagerly or into the capture that is open.

        ``after_solver_loop`` is called once per microstep, right after the loop of its
        constraint solve has been issued (``RigidStep.step``). ``first_state_prepared``
        says that ``prepare_trials`` has been issued in the same capture for this start
        state, controls and interval.
        """
        self._validate_execution_context()
        self._validate_trial_arguments(state_in, state_out, control, dt)
        if contacts is not None or observables is not None:
            raise ValueError("Prototype exports rigid state only; inspect mjw_data outside timed trials")
        if self.rigid_step is None:
            # MuJoCo Warp belongs to the examples extra; the package imports without it.
            from newton_solver_suturebot._src.solvers.mujoco.warp_step import RigidStep, companion_data

            # Refused during capture: the first interval runs eagerly.
            self.rigid_step = RigidStep(self.mjw_model, self.mjw_data)
            if self.rigid_substeps > 1:
                self._first_step = RigidStep(self.mjw_model, companion_data(self.mjw_data))
                self._prepared_overflow = wp.zeros_like(self.mjw_data.overflow)
        self.rigid_step.validate(self.mjw_model, self.mjw_data)
        if first_state_prepared and self._first_step is None:
            raise RuntimeError("No first microstep was prepared (prepare_trials returned False)")
        self._trial_dt = float(dt)
        try:
            self._issue_microsteps(state_in, state_out, control, dt, after_solver_loop, first_state_prepared)
        except BaseException:
            # Never retain a partly populated first warmup as a complete contract.
            if not self._rigid_warmed and not self.model.device.is_capturing:
                self.rigid_step = self._first_step = self._prepared_overflow = None
            raise
        if not self.model.device.is_capturing:
            self._rigid_warmed = True

    def prepare_trials(self, state_in, control, dt):
        """Issue once what every trial of one interval repeats in its first microstep.

        Trials of one interval start from the same position, velocity, activation and
        controls and differ in the applied wrenches. The stages of the first microstep
        that read only that state (``RigidStep.state_stages``) are therefore the same in
        every trial. This issues them once, on the Data of the first microstep, and keeps
        the overflow flags they raise; ``_step_uncaptured(..., first_state_prepared=True)``
        then issues the rest of the interval.

        Issue it in the capture that holds the trials, after restoring the checkpoint
        they start from.

        Returns:
            False if an interval has one microstep and nothing is shared; the trials then
            run every stage themselves.
        """
        self._validate_execution_context()
        if self._first_step is None:
            if self.rigid_step is None:
                raise RuntimeError("Warm the rigid step outside CUDA capture before preparing trials")
            return False
        with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
            self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
            self._update_mjc_data(self.mjw_data, self.model, state_in)
            # the state stages read the microstep (constraint rows, passive forces)
            self._write_microstep(dt)
            self.mjw_data.overflow.zero_()
            self._first_step.state_stages()
            wp.copy(self._prepared_overflow, self.mjw_data.overflow)
        return True

    def _issue_microsteps(self, state_in, state_out, control, dt, after_solver_loop, first_state_prepared):
        with wp.ScopedDevice(self.model.device), self._scoped_mujoco_warp_execution():
            self._enable_rne_postconstraint(state_out, None)
            self._apply_mjc_control(self.model, state_in, control, self.mjw_data)
            self._update_mjc_data(self.mjw_data, self.model, state_in)
            self._write_microstep(dt)
            # Accumulate overflow over every microstep; never discard evidence
            # that an intermediate microstep truncated contacts/constraints.
            if first_state_prepared:
                # start from the flags that the prepared stages of the first microstep raised
                wp.copy(self.mjw_data.overflow, self._prepared_overflow)
            else:
                self.mjw_data.overflow.zero_()
            # The first microstep runs on its own Data, which shares the integration state
            # with the Data of the others. What later code reads from mjw_data (contacts,
            # constraint rows, forces) is therefore what the last microstep left there.
            steps = [self._first_step or self.rigid_step] + [self.rigid_step] * (self.rigid_substeps - 1)
            stages = [not first_state_prepared] + [True] * (self.rigid_substeps - 1)
            if not getattr(self, "trial_warm_start", False):
                for rigid_step, state_stages in zip(steps, stages, strict=True):
                    rigid_step.step(after_solver_loop, state_stages=state_stages)
            else:
                warm_start = self.mjw_data.qacc_warmstart
                if self._trial_warm_slots is None:
                    if self.model.device.is_capturing:
                        raise RuntimeError("Warm the rigid trial storage before CUDA capture")
                    self._trial_warm_slots = tuple(
                        wp.zeros_like(warm_start, requires_grad=False) for _ in range(self.rigid_substeps)
                    )
                    self._trial_warm_valid = wp.zeros(1, dtype=int, device=self.model.device)
                slots, valid = self._trial_warm_slots, self._trial_warm_valid
                wp.launch(_load_trial_warm_start, warm_start.shape, [valid, slots[0], warm_start])
                for index, (rigid_step, state_stages) in enumerate(zip(steps, stages, strict=True)):
                    rigid_step.step(after_solver_loop, state_stages=state_stages)
                    if index + 1 < self.rigid_substeps:
                        wp.launch(
                            _exchange_trial_warm_start,
                            warm_start.shape,
                            [valid, warm_start, slots[index], slots[index + 1]],
                        )
                    else:
                        wp.launch(_store_trial_warm_start, warm_start.shape, [warm_start, slots[index], valid])
            self._update_newton_state(self.model, state_out, self.mjw_data, state_prev=state_in)
