"""Cache immutable-start FEM force assembly; trial loads/CG remain unchanged."""

import math
from collections import OrderedDict

import warp as wp

from ...utils.capture import ScopedCapture
from .differentiation import active_tape
from .implicit_predictor import restore_spring_damping_predictor
from .prepared_step import _predictor, _subtract

# Solver methods whose original implementation a held start stands for.
_ORIGINAL_METHODS = (
    "step",
    "implicit_integration",
    "_accumulate_forces",
    "eval_spring_forces",
    "eval_tetrahedral_forces",
    "eval_hexahedral_forces",
    "eval_triangle_forces",
    "eval_particle_ground_contact_forces",
    "eval_gravity_forces",
    "eval_linear_damping_forces",
)
# Force scratch that SolverFEM._accumulate_forces clears and sums for the supported terms.
_FORCE_TERMS = ("surface", "spring", "tet", "hex", "tri", "ground", "gravity", "damping")
# Start states with a kept preparation graph. Two accepted states alternate; a graph serves every
# timestep, which it reads from the solver's step time.
_GRAPH_LIMIT = 8


@wp.kernel(enable_backward=False)
def copy_impulse(source: wp.array[wp.vec3], target: wp.array[wp.vec3], uses: wp.array[int]):
    i = wp.tid()
    target[i] = source[i]
    if i == 0:
        uses[0] += 1


def storage(array):
    return array.ptr, array.shape, array.strides, array.dtype, array.device.alias


class HeldFEMStart:
    """Owned force impulse and elastic predictor for a single held-start attempt.

    prepare() runs once per attempt, outside every capture of the trials. All
    subsequent loads preserve the original order: cached h*sum(forces), then
    h*external, then predictor subtraction, then spring damping correction. The
    ordinary compact solver still normalizes, solves, reconstructs and restores
    fixed boundaries.

    On a CUDA device the preparation of a start state is issued as written
    once and replayed as a CUDA graph from its second attempt on. A graph
    stands for one binding: the storage of every array the preparation reads
    or writes, its launch sizes and its scalar arguments. Values written in
    place between attempts, such as the accepted state, the refreshed tangent,
    a topology edit within fixed storage or the timestep, which is device
    data (``solver.step_time``), are read by every replay. Any other binding
    is prepared as written again before it is captured.
    """

    def __init__(self, solver):
        self.solver = solver
        self.device = solver.model.device
        self.state = solver.model.state()
        self.predictor = wp.zeros_like(self.state.particle_f)
        self.uses = wp.zeros(1, dtype=int, device=self.device)
        self.active = False
        self.signature = None
        self.preparations = 0
        self.status = "not prepared"
        # Start state -> None after its eager preparation, then the graph that replays it.
        self._graphs = OrderedDict()
        # What every kept graph bakes in besides its start state.
        self._graph_resources = None
        self.graph_captures = self.graph_launches = 0

    def supported(self, state_in, control, contacts):
        from .solver import SolverFEM

        s = self.solver
        if type(s) is not SolverFEM or any(
            getattr(getattr(s, name), "__func__", None) is not getattr(SolverFEM, name) for name in _ORIGINAL_METHODS
        ):
            return False
        if (
            s.differentiable
            or active_tape() is not None
            or getattr(s.model, "requires_grad", False)
            or s._streaming is not None
            or s._component_scaling is not None
            or s._tangent_refresh_captured
            or s.soft_surface_contacts is not None
            or s._ground_plane is not None
            or contacts is not None
            or control is None
        ):
            return False
        if any(a.requires_grad for a in (state_in.particle_q, state_in.particle_qd, state_in.particle_f)):
            return False
        if any(
            getattr(getattr(control, name, None), "requires_grad", False)
            for name in ("joint_f", "joint_target_q", "joint_target_qd", "tet_activations")
        ):
            return False
        return True

    def prepare(self, state_in, control, contacts, dt):
        self.active = False
        self.signature = None
        if not self.supported(state_in, control, contacts):
            self.status = "unsupported held-start configuration"
            return False
        if self.device.is_capturing:
            raise RuntimeError("Held-start preparation must run before CUDA graph capture")
        if not math.isfinite(dt) or dt <= 0 or dt != self.solver.dt:
            raise ValueError("Held-start timestep must match the assembled FEM tangent")
        s = self.solver
        s.validate_observables(None, contacts)
        # The preparation reads the timestep from the solver's step time, eagerly,
        # while it is captured and in every replay: write it before any of them.
        s.step_time.set(dt)
        binding = self._binding(state_in, control) if self.device.is_cuda else None
        if binding not in self._graphs:
            self._evaluate(state_in, control, contacts, dt)
            if binding is not None:
                # The kernels are loaded and every scratch array exists: the next preparation
                # of this binding is captured without an allocation.
                self._graphs[binding] = None
                if len(self._graphs) > _GRAPH_LIMIT:
                    self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(binding)
            graph = self._graphs[binding]
            if graph is None:
                with ScopedCapture(device=self.device) as capture:
                    self._evaluate(state_in, control, contacts, dt)
                graph = self._graphs[binding] = capture.graph
                self.graph_captures += 1
            wp.capture_launch(graph)
            self.graph_launches += 1
        self.signature = (
            storage(state_in.particle_q),
            storage(state_in.particle_qd),
            storage(state_in.particle_f),
            id(control),
            float(dt),
            s._differentiation_revision,
            s._tangent_version,
        )
        self.active = True
        self.preparations += 1
        self.status = "prepared held-start force impulse and predictor"
        return True

    def warm(self, state_in, control, contacts, dt):
        """Capture the preparation graph of ``state_in`` ahead of its first attempt and launch it once.

        For start-up, on a CUDA device and outside capture. :meth:`prepare` issues the
        preparation of a start state as written at its first attempt and captures it at
        the second; this entry issues both now, so that no attempt of the run captures.
        The evaluation as written comes first because it loads the kernels and makes any
        force scratch of the solver that does not exist yet, without which the
        preparation has no binding. What the launches write is this object's own state
        copy and predictor and the solver's force scratch, all of which the next
        :meth:`prepare` overwrites.
        ``active``, ``signature``, ``preparations`` and ``graph_launches`` are untouched:
        they belong to the attempts.

        Returns:
            True if the start state has its graph after the call; False for a device
            without capture or a configuration that :meth:`prepare` does not serve.
        """
        if not self.device.is_cuda or not self.supported(state_in, control, contacts):
            return False
        if self.device.is_capturing:
            raise RuntimeError("Held-start preparation must run before CUDA graph capture")
        if not math.isfinite(dt) or dt <= 0 or dt != self.solver.dt:
            raise ValueError("Held-start timestep must match the assembled FEM tangent")
        s = self.solver
        s.validate_observables(None, contacts)
        # Every launch below reads the timestep from the solver's step time.
        s.step_time.set(dt)
        binding = self._binding(state_in, control)
        if binding is None or binding not in self._graphs:
            # As prepare() does at the first attempt of a binding. No binding exists while
            # force scratch of the solver is missing; this evaluation makes it.
            self._evaluate(state_in, control, contacts, dt)
            binding = self._binding(state_in, control)
            if binding is None:
                return False
            self._graphs[binding] = None
            if len(self._graphs) > _GRAPH_LIMIT:
                self._graphs.popitem(last=False)
        if self._graphs[binding] is None:
            with ScopedCapture(device=self.device) as capture:
                self._evaluate(state_in, control, contacts, dt)
            self._graphs[binding] = capture.graph
            self.graph_captures += 1
            # The executable graph is built at the first launch: here, not in an attempt.
            wp.capture_launch(capture.graph)
        return True

    def _evaluate(self, state_in, control, contacts, dt):
        """Issue the force impulse and the elastic predictor of the start state."""
        s = self.solver
        # Only particle q/qd are inputs of the supported force evaluators.
        wp.copy(self.state.particle_q, state_in.particle_q)
        wp.copy(self.state.particle_qd, state_in.particle_qd)
        if s._fixed_particles is not None:
            s._fixed_particles.restore(self.state)
        s._accumulate_forces(s.model, self.state, contacts, control, dt)
        if s.model.tet_count or s.hex_count or s.model.spring_count:
            matrix = s.A_bsr
            wp.launch(
                _predictor,
                s.model.particle_count,
                [matrix.offsets, matrix.columns, matrix.values, self.state.particle_qd, self.predictor],
                device=self.device,
            )
        else:
            self.predictor.zero_()

    def _binding(self, state_in, control):
        """Return the start state that selects a preparation graph, or None without one.

        Call after supported() accepts the configuration. A graph holds the
        storage of every array of its launches, the launch sizes and the scalar
        arguments. Everything but the start state is common to the kept graphs,
        which are dropped when any of it is replaced. The timestep is none of
        these: the launches read it from ``solver.step_time``, which prepare()
        writes before the preparation is issued, captured or replayed.
        """
        s = self.solver
        model, matrix, fixed = s.model, s.A_bsr, s._fixed_particles
        forces = tuple(map(s._force_bufs.get, _FORCE_TERMS))
        if any(force is None for force in forces):
            # The eager preparation allocates the missing scratch.
            return None
        arrays = (
            *self.owners[:4],
            *forces,
            s._get_gravity_array(model),
            model.particle_mass,
            model.particle_flags,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            control.tet_activations if s.enable_tet_activations else None,
            s.hex_indices,
            s.hex_inv_J0,
            s.hex_det_J0_w,
            s.hex_materials,
            model.tri_indices,
            model.tri_poses,
            model.tri_materials,
            model.spring_indices,
            model.spring_rest_length,
            model.spring_stiffness,
            model.spring_damping,
            s.spring_rest_direction,
            matrix.offsets,
            matrix.columns,
            matrix.values,
            None if fixed is None else fixed.indices,
            None if fixed is None else fixed.reference,
        )
        resources = (
            tuple(None if array is None else storage(array) for array in arrays),
            # Launch sizes and the branches that follow from them.
            (model.particle_count, model.tet_count, s.hex_count, model.tri_count, model.spring_count),
            # Scalar kernel arguments and the branches that follow from them.
            (s.min_stretch, s.max_stretch, s.linear_damping, s.torque_stiffness, s.torque_damping),
            s.enable_tet_activations,
            # A replaced force implementation issues other launches.
            tuple(getattr(type(s), name) for name in _ORIGINAL_METHODS),
        )
        if resources != self._graph_resources:
            # Storage, sizes, force laws or the tangent's arrays are replaced: no kept graph addresses them.
            self._graphs.clear()
            self._graph_resources = resources
        return storage(state_in.particle_q), storage(state_in.particle_qd)

    def matches(self, state, dt, control=None):
        if not self.active or self.signature is None:
            return False
        current = (storage(state.particle_q), storage(state.particle_qd), storage(state.particle_f))
        return (
            current == self.signature[:3]
            and (control is None or id(control) == self.signature[3])
            and float(dt) == self.signature[4]
            and self.solver._differentiation_revision == self.signature[5]
            and self.solver._tangent_version == self.signature[6]
        )

    def restore_impulse(self, target):
        wp.launch(
            copy_impulse,
            self.solver.model.particle_count,
            [self.state.particle_f, target, self.uses],
            device=self.device,
        )

    def subtract_predictor(self, target, dt):
        wp.launch(_subtract, self.solver.model.particle_count, [self.predictor, target], device=self.device)
        if self.solver.model.spring_count:
            model = self.solver.model
            self.solver.step_time.set(dt)
            wp.launch(
                restore_spring_damping_predictor,
                model.spring_count,
                [
                    model.spring_indices,
                    model.spring_damping,
                    self.solver.step_time.seconds,
                    self.state.particle_qd,
                    target,
                ],
                device=self.device,
            )

    def invalidate(self):
        self.active = False
        self.signature = None
        self.status = "inactive outside an explicitly prepared attempt"

    @property
    def owners(self):
        return self.state.particle_q, self.state.particle_qd, self.state.particle_f, self.predictor, self.uses

    def capture_key(self):
        current = (
            self.active
            and self.signature is not None
            and self.solver._differentiation_revision == self.signature[5]
            and self.solver._tangent_version == self.signature[6]
        )
        return None if not current else (id(self), tuple(storage(a) for a in self.owners))

    def diagnostics(self):
        return {
            "preparations": self.preparations,
            "cached_trial_assemblies": int(self.uses.numpy()[0]),
            "status": self.status,
        }
