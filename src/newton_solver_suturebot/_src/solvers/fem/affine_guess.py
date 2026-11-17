# SPDX-License-Identifier: Apache-2.0
"""Guarded stale-response CG guess; physical equations are unchanged.

Only an initial numerical guess is selected. Every trial still executes the
original current-matrix CG, stopping rule, iteration budget and reconstruction.
The response provider is injected and runs outside capture at support changes.
"""

from typing import Any

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def initialize_flags(history: wp.array[int], maximum: wp.array[float], flags: wp.array[int]):
    # First prototype deliberately falls back for tiny-RHS normalization. This
    # avoids comparing different scaled coordinates; all default measured S=1.
    flags[0] = history[0]
    flags[1] = 0
    flags[2] = 0
    flags[3] = int(maximum[0] > 0.0 and maximum[0] < 1.0e-12)
    flags[4] = 0


@wp.kernel(enable_backward=False)
def delta_coefficients(
    free: wp.array[int],
    support: wp.array[int],
    count: wp.array[int],
    physical_rhs: wp.array[wp.vec3],
    previous_rhs: wp.array[wp.vec3],
    coefficients: wp.array[float],
):
    column = wp.tid()
    value = float(0.0)
    if column < count[0]:
        dof = support[column]
        node, axis = dof // 3, dof % 3
        value = physical_rhs[free[node]][axis] - previous_rhs[node][axis]
    coefficients[column] = value


@wp.kernel(enable_backward=False)
def propose(
    basis: wp.array[float],
    count: wp.array[int],
    coefficients: wp.array[float],
    original: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    flags: wp.array[int],
    candidate: wp.array[wp.vec3],
):
    node = wp.tid()
    value = original[node]
    if flags[0] != 0 and flags[3] == 0:
        for axis in range(3):
            if value[axis] != previous[node][axis]:
                wp.atomic_max(flags, 1, 1)
        correction = wp.vec3(0.0)
        stride = 3 * original.shape[0]
        for column in range(count[0]):
            offset = column * stride + 3 * node
            correction += coefficients[column] * wp.vec3(basis[offset], basis[offset + 1], basis[offset + 2])
        value += correction
    for axis in range(3):
        if not wp.isfinite(value[axis]):
            wp.atomic_max(flags, 2, 1)
    candidate[node] = value


_BLOCK = 256


@wp.func
def _square(value: float):
    return wp.float64(value) * wp.float64(value)


@wp.kernel(enable_backward=False)
def norm_partials(
    original: wp.array[float],
    candidate: wp.array[float],
    partials: wp.array2d[wp.float64],
):
    block, _lane = wp.tid()
    a = wp.tile_load(original, shape=_BLOCK, offset=block * _BLOCK)
    b = wp.tile_load(candidate, shape=_BLOCK, offset=block * _BLOCK)
    wp.tile_store(partials[0], wp.tile_sum(wp.tile_map(_square, a)), offset=block)
    wp.tile_store(partials[1], wp.tile_sum(wp.tile_map(_square, b)), offset=block)


@wp.kernel(enable_backward=False)
def reduce_norms(partials: wp.array2d[wp.float64], norms: wp.array[wp.float64]):
    a, b = wp.float64(0.0), wp.float64(0.0)
    for block in range(partials.shape[1]):
        a += partials[0, block]
        b += partials[1, block]
    norms[0] = a
    norms[1] = b


@wp.kernel(enable_backward=False)
def cpu_norms(
    original: wp.array[wp.vec3],
    candidate: wp.array[wp.vec3],
    norms: wp.array[wp.float64],
):
    a, b = wp.float64(0.0), wp.float64(0.0)
    for node in range(original.shape[0]):
        for axis in range(3):
            av, bv = wp.float64(original[node][axis]), wp.float64(candidate[node][axis])
            a += av * av
            b += bv * bv
    norms[0] = a
    norms[1] = b


@wp.kernel(enable_backward=False)
def choose(flags: wp.array[int], norms: wp.array[wp.float64], counters: wp.array[int]):
    counters[0] += 1
    if flags[0] == 0:
        counters[2] += 1
    elif flags[3] != 0:
        counters[3] += 1
    elif flags[1] != 0:
        counters[4] += 1
    elif flags[2] != 0 or not wp.isfinite(norms[0]) or not wp.isfinite(norms[1]):
        counters[5] += 1
    elif norms[1] < norms[0]:
        flags[4] = 1
        counters[1] += 1
    else:
        counters[6] += 1


@wp.kernel(enable_backward=False)
def select_guess(flags: wp.array[int], candidate: wp.array[wp.vec3], guess: wp.array[wp.vec3]):
    node = wp.tid()
    if flags[4] != 0:
        guess[node] = candidate[node]


@wp.kernel(enable_backward=False)
def save_physical_history(
    free: wp.array[int],
    rhs: wp.array[wp.vec3],
    normalized_increment: wp.array[wp.vec3],
    maximum: wp.array[float],
    previous_rhs: wp.array[wp.vec3],
    previous_increment: wp.array[wp.vec3],
):
    node = wp.tid()
    scale = float(1.0)
    if maximum[0] > 0.0 and maximum[0] < 1.0e-12:
        scale = wp.min(1.0 / maximum[0], 1.0e30)
    previous_rhs[node] = rhs[free[node]]
    # Same physical conversion performed by restore_linear_increment after
    # the ordinary scatter; history never stores a differently scaled RHS/x.
    previous_increment[node] = normalized_increment[node] / scale


@wp.kernel(enable_backward=False)
def finish_statistics(
    iterations: wp.array[int],
    recursive: wp.array(dtype=Any),
    threshold: wp.array(dtype=Any),
    maxiter: int,
    history: wp.array[int],
    counters: wp.array[int],
):
    history[0] = 1
    counters[7] += iterations[0]
    counters[8] += int(iterations[0] >= maxiter)
    counters[9] += int(not wp.isfinite(recursive[0]) or recursive[0] > threshold[0])


# The two argument types the solver's workspaces pass (plain CG: float32, certified
# solve: float64), declared so that the module has its final hash when it is imported:
# a first launch then adds no overload and loads no second build of the module.
for _scalar in (wp.float32, wp.float64):
    wp.overload(finish_statistics, {"recursive": wp.array(dtype=_scalar), "threshold": wp.array(dtype=_scalar)})
del _scalar


COUNTERS = (
    "fresh_trials",
    "selected",
    "no_history",
    "scaled_rhs_fallback",
    "history_mismatch",
    "nonfinite",
    "no_improvement",
    "cg_iterations",
    "cg_cap_hits",
    "recursive_failures",
)


class DeviceBasisGuess:
    """Persistent graph-safe storage and kernels, also exercised on CPU."""

    def __init__(self, operator, free, *, capacity=96):
        self.operator, self.free = operator, free
        self.device, self.n = free.device, free.size
        if capacity <= 0 or free.size <= 0:
            raise ValueError("Response capacity and free-node count must be positive")
        self.capacity = capacity
        self.basis = wp.zeros(3 * self.n * capacity, dtype=float, device=self.device)
        self.support = wp.zeros(capacity, dtype=int, device=self.device)
        self.count = wp.zeros(1, dtype=int, device=self.device)
        self.coefficients = wp.zeros(capacity, dtype=float, device=self.device)
        self.previous_rhs = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.previous_increment = wp.zeros_like(self.previous_rhs)
        self.candidate = wp.zeros_like(self.previous_rhs)
        self.original_residual = wp.zeros_like(self.previous_rhs)
        self.candidate_residual = wp.zeros_like(self.previous_rhs)
        self.norms = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.blocks = (3 * self.n + _BLOCK - 1) // _BLOCK
        self.partials = wp.zeros((2, self.blocks), dtype=wp.float64, device=self.device)
        self.original_scalar = wp.array(
            ptr=self.original_residual.ptr,
            shape=(3 * self.n,),
            dtype=float,
            device=self.device,
        )
        self.candidate_scalar = wp.array(
            ptr=self.candidate_residual.ptr,
            shape=(3 * self.n,),
            dtype=float,
            device=self.device,
        )
        self.flags = wp.zeros(5, dtype=int, device=self.device)
        self.history = wp.zeros(1, dtype=int, device=self.device)
        self.counters = wp.zeros(len(COUNTERS), dtype=int, device=self.device)
        # Nonzero while the manager applies this basis. A captured solve issues ``before``
        # and ``after`` behind this flag, so one graph serves an applied and a withheld basis.
        self.enabled = wp.zeros(1, dtype=int, device=self.device)
        self._enabled = False

    @property
    def owners(self):
        return tuple(value for value in vars(self).values() if isinstance(value, wp.array))

    def set_enabled(self, value):
        """Write the device flag where it changes; nothing is read from the device."""
        value = bool(value)
        if value == self._enabled:
            return
        if self.device.is_capturing:
            raise RuntimeError("Switch the response-basis guess outside CUDA graph capture")
        self.enabled.fill_(int(value))
        self._enabled = value

    def load_basis(self, columns, support):
        support = np.asarray(support, dtype=np.int32)
        if support.ndim != 1 or not len(support) or len(support) > self.capacity:
            raise ValueError("Support exceeds fixed response capacity")
        if len(np.unique(support)) != len(support) or np.any(support < 0) or np.any(support >= 3 * self.n):
            raise ValueError("Invalid response support")
        if (
            columns.dtype != wp.float32
            or columns.shape != (3 * self.n * len(support),)
            or columns.device != self.device
        ):
            raise ValueError("Expected flat column-major FP32 response storage")
        self.count.zero_()
        wp.copy(self.basis, columns, count=columns.size)
        padded = np.zeros(self.capacity, dtype=np.int32)
        padded[: len(support)] = support
        self.support.assign(padded)
        self.count.fill_(len(support))
        self.history.zero_()

    def before(self, physical_rhs, normalized_rhs, normalized_guess, maximum):
        self.norms.zero_()
        wp.launch(initialize_flags, 1, [self.history, maximum, self.flags], device=self.device)
        wp.launch(
            delta_coefficients,
            self.capacity,
            [
                self.free,
                self.support,
                self.count,
                physical_rhs,
                self.previous_rhs,
                self.coefficients,
            ],
            device=self.device,
        )
        wp.launch(
            propose,
            self.n,
            [
                self.basis,
                self.count,
                self.coefficients,
                normalized_guess,
                self.previous_increment,
                self.flags,
                self.candidate,
            ],
            device=self.device,
        )
        self.operator.matvec(
            normalized_guess,
            normalized_rhs,
            self.original_residual,
            alpha=-1.0,
            beta=1.0,
        )
        self.operator.matvec(
            self.candidate,
            normalized_rhs,
            self.candidate_residual,
            alpha=-1.0,
            beta=1.0,
        )
        if self.device.is_cuda:
            wp.launch(
                norm_partials,
                (self.blocks, _BLOCK),
                [self.original_scalar, self.candidate_scalar, self.partials],
                block_dim=_BLOCK,
                device=self.device,
            )
            wp.launch(reduce_norms, 1, [self.partials, self.norms], device=self.device)
        else:
            wp.launch(
                cpu_norms,
                1,
                [self.original_residual, self.candidate_residual, self.norms],
                device=self.device,
            )
        wp.launch(choose, 1, [self.flags, self.norms, self.counters], device=self.device)
        wp.launch(
            select_guess,
            self.n,
            [self.flags, self.candidate, normalized_guess],
            device=self.device,
        )

    def after(self, physical_rhs, normalized_increment, maximum, linear):
        wp.launch(
            save_physical_history,
            self.n,
            [
                self.free,
                physical_rhs,
                normalized_increment,
                maximum,
                self.previous_rhs,
                self.previous_increment,
            ],
            device=self.device,
        )
        wp.launch(
            finish_statistics,
            1,
            [
                linear._cur_iter_and_condition,
                linear._tiled_dot.col(0),
                linear._residuals[1],
                linear._maxiter,
                self.history,
                self.counters,
            ],
            device=self.device,
        )

    def reset(self):
        self.history.zero_()


class AffineGuessManager:
    """Bind a numerical response once track support exists; never replace A.

    ``nodes_per_attempt`` (default ``None``: off) bounds the response columns that one
    attempt solves when the track support has grown: at most the columns of that many
    support nodes (three columns each). A track sample that enters a new tetrahedron
    next to the last one adds one node; the first sample of a new section adds the four
    nodes of its tetrahedron, whose twelve columns are otherwise solved in the one
    attempt that follows the entry. With the bound the missing columns are solved over
    the following attempts, the provider keeping what it has, and the basis is loaded
    and applied again in the attempt that solves the last of them. Until then the guess
    is withheld (``status`` "solving response columns"), as it is before the first track
    support. Every column is the provider's solve of the same unit load on the same
    reference matrix as in an unbounded build, so the loaded basis is the same
    (``tests/test_response_basis_bound.py`` compares the columns of the Warp provider bit
    for bit). ``deferred_builds`` counts the attempts that solved columns without
    loading them. A structure change, an expired basis and the first core are built at
    once, as without the bound.

    An attempt that solves without loading asks the provider to extend what it holds
    (``provider.extend``, where the provider has it): the Warp provider writes the
    columns into its fixed storage and allocates and reads nothing. The attempt that
    loads the basis makes the provider's one request, with its one read of the residual
    packets of every column solved since the last load, so a bounded build reads the
    device as often as an unbounded one.
    """

    def __init__(self, solver, provider, *, capacity=96, refresh_generations=0, nodes_per_attempt=None):
        if nodes_per_attempt is not None and (
            isinstance(nodes_per_attempt, bool)
            or not isinstance(nodes_per_attempt, (int, np.integer))
            or nodes_per_attempt < 1
        ):
            raise ValueError("nodes_per_attempt must be None or a positive integer")
        self.solver, self.provider = solver, provider
        self.capacity, self.refresh_generations = capacity, refresh_generations
        self.nodes_per_attempt = None if nodes_per_attempt is None else int(nodes_per_attempt)
        # Support whose columns the provider holds while a bounded build is under way (None: the
        # provider holds the columns of ``support``), and the attempts that solved without loading.
        self._solved = None
        self.deferred_builds = 0
        self.core = None
        self._active = False
        self.support = ()
        self.structural_key = None
        self.basis_generation = None
        self.basis_builds = 0
        self.preparation_reports = []
        self.status = "waiting for track support"

    @property
    def active(self):
        """Whether the basis is applied to the solves of the current attempt."""
        return self._active

    @active.setter
    def active(self, value):
        self._active = bool(value)
        if self.core is not None:
            # The same decision as device data, for the solves that a graph holds.
            self.core.set_enabled(self._active)

    @property
    def _gated(self):
        """Whether a captured solve takes the basis by the device flag of its core.

        With conditional CUDA graphs a graph holds the launches of the basis behind
        ``core.enabled`` and its key does not follow ``active``. Elsewhere a graph
        holds them only while the manager is active, and its key says which.
        """
        return self.solver.model.device.is_cuda and wp.is_conditional_graph_supported()

    def _structure(self):
        fixed = self.solver._fixed_particles
        matrix = None if fixed is None else fixed.compact_matrix
        if matrix is None:
            return None
        return (
            id(fixed),
            id(matrix),
            matrix.nrow,
            matrix.ncol,
            matrix.nnz,
            matrix.offsets.ptr,
            matrix.columns.ptr,
            matrix.values.ptr,
            id(fixed.compact_operator),
            fixed.free.ptr,
            fixed.free.size,
        )

    def _supported(self, coupler, state_in, control):
        """Host tests of the solver configuration that this guess serves; they read no device data."""
        s = self.solver
        from ..mujoco.experimental_rigid_backend import supported_subcycled
        from .differentiation import active_tape
        from .solver import SolverFEM

        return not (
            type(s) is not SolverFEM
            or getattr(s.step, "__func__", None) is not SolverFEM.step
            or not supported_subcycled(coupler.rigid_solver)
            or s.model.body_count <= 1
            or control is None
            or any(
                getattr(getattr(owner, name, None), "requires_grad", False)
                for owner in (state_in, getattr(coupler, "_candidate", None))
                for name in (
                    "particle_q",
                    "particle_qd",
                    "particle_f",
                    "joint_q",
                    "joint_qd",
                    "body_q",
                    "body_qd",
                    "body_f",
                )
            )
            or any(
                getattr(getattr(control, name, None), "requires_grad", False)
                for name in ("joint_f", "joint_target_q", "joint_target_qd", "tet_activations")
            )
            or getattr(getattr(getattr(control, "mujoco", None), "ctrl", None), "requires_grad", False)
            or s.velocity_increment.requires_grad
            or not s.model.device.is_cuda
            or s.differentiable
            or active_tape() is not None
            or getattr(s.model, "requires_grad", False)
            or s._streaming is not None
            or s._component_scaling is not None
            or s.solver_type != "cg"
            or s.linear_solver_check_every != 0
            or s.preconditioner_type != "block_diag"
            or s._tangent_refresh_captured
        )

    def prepare(self, coupler, state_in, control):
        """Make the storage of the basis and the column engine ahead of the first track support.

        For start-up, outside capture and before :meth:`begin_attempt` of the same
        attempt. For a supported configuration with a compact free operator it creates
        the core (empty, its device flag 0) and the provider's engine and records the
        structure they belong to. The first track support then loads its columns into
        this core: it replaces no storage, so a graph captured before it stays valid.
        Nothing is done when a core of the current structure exists. No basis is built
        and the manager stays inactive.

        Returns:
            True if a core of the current structure exists after the call.
        """
        if self.solver.model.device.is_capturing:
            raise RuntimeError("Prepare the response-basis storage outside CUDA graph capture")
        if not self._supported(coupler, state_in, control):
            return False
        coupler._refresh_projection()
        structure = self._structure()
        if structure is None:
            return False
        if self.core is not None and structure == self.structural_key:
            return True
        fixed = self.solver._fixed_particles
        if self.core is not None:
            # A replaced structure, as at a support change: no graph addresses the new storage.
            coupler.invalidate_fem_graph()
        if hasattr(self.provider, "invalidate"):
            self.provider.invalidate()
        # The provider holds no column now: a bounded build that was under way starts over.
        self._solved = None
        if hasattr(self.provider, "prepare"):
            self.provider.prepare(fixed.compact_matrix)
        self.active = False
        self.core = DeviceBasisGuess(fixed.compact_operator, fixed.free, capacity=self.capacity)
        self.support, self.structural_key = (), structure
        self.basis_generation = None
        self._track_count = None
        return True

    def begin_attempt(self, coupler, state_in, control):
        self.active = False
        if self.core is not None:
            self.core.reset()
        s = self.solver
        if not self._supported(coupler, state_in, control):
            self.status = "unsupported solver configuration"
            return
        interaction = coupler.interaction
        if interaction.track_count <= 0:
            self.status = "waiting for committed track support"
            return
        coupler._refresh_projection()
        structure = self._structure()
        if structure is None:
            self.status = "missing compact free operator"
            return
        fixed = s._fixed_particles
        # Track count changes only at committed material events. The free map is the
        # host array that the fixed-particle operator was built from: nothing is read.
        count = interaction.track_count
        need_support = getattr(self, "_track_count", None) != count or structure != self.structural_key
        requested = self.support
        if need_support:
            free = getattr(fixed, "free_host", None)
            if free is None:
                free = fixed.free.numpy()
            inverse = {int(node): index for index, node in enumerate(free)}
            tracks = np.asarray(interaction._track_nodes_host[:count], dtype=np.int32).reshape(-1)
            nodes = sorted({inverse[int(node)] for node in tracks if int(node) in inverse})
            requested = tuple(3 * node + axis for node in nodes for axis in range(3))
        if not requested or len(requested) > self.capacity:
            self.status = "track support outside response capacity"
            return
        changed = structure != self.structural_key
        expired = (
            self.refresh_generations > 0
            and self.basis_generation is not None
            and s._tangent_version - self.basis_generation >= self.refresh_generations
        )
        if changed or tuple(requested) != self.support or expired or self.core is None:
            if changed and self.core is not None:
                coupler.invalidate_fem_graph()
            if (changed or expired) and hasattr(self.provider, "invalidate"):
                self.provider.invalidate()
            if changed or expired or self.core is None:
                self._solved = None
            elif self.nodes_per_attempt is not None:
                # The provider holds the columns of the support it was last asked for. Of the
                # nodes it has no column of, this attempt solves the first ones; the others wait.
                held = set(self.support if self._solved is None else self._solved)
                missing = list(dict.fromkeys(dof // 3 for dof in requested if dof not in held))
                if len(missing) > self.nodes_per_attempt:
                    solve = set(missing[: self.nodes_per_attempt])
                    partial = tuple(dof for dof in requested if dof in held or dof // 3 in solve)
                    extend = getattr(self.provider, "extend", None)
                    if extend is None:
                        # A provider without storage of its own for columns that are not yet loaded.
                        self.provider(fixed.compact_matrix, partial)
                    else:
                        extend(fixed.compact_matrix, partial)
                    self._solved = partial
                    self.deferred_builds += 1
                    # Host numbers only: the columns are diagnosed by the request that loads them.
                    self.preparation_reports.append(
                        {
                            "generation": s._tangent_version,
                            "support_dofs": len(partial),
                            "new_columns": len(partial) - len(held.intersection(partial)),
                            "loaded": False,
                        }
                    )
                    # The track count is not recorded: the next attempt asks for the support again.
                    self.status = "solving response columns"
                    return
            columns, metadata = self.provider(fixed.compact_matrix, requested)
            self._solved = None
            if self.core is None or changed:
                self.core = DeviceBasisGuess(fixed.compact_operator, fixed.free, capacity=self.capacity)
            self.core.load_basis(columns, requested)
            self.support, self.structural_key = tuple(requested), structure
            self.basis_generation = s._tangent_version
            self.basis_builds += 1
            self.preparation_reports.append(
                metadata | {"generation": s._tangent_version, "support_dofs": len(requested)}
            )
        self._track_count = count
        self.active, self.status = True, "guarded stale-basis guess"

    def reset(self):
        if self.core is not None:
            self.core.reset()

    def end_attempt(self):
        """Keep the owned basis, but never apply it outside its coupling attempt."""
        self.active = False
        self.reset()

    def invalidate(self):
        self.active = False
        self.structural_key = None
        self.reset()

    def capture_key(self):
        """What a captured solve bakes in of this guess, or None while it holds no launch of it.

        With conditional CUDA graphs: the storage of the core, from the moment a core
        exists; whether the basis is applied is the core's device flag, which is in no
        key. Elsewhere: that storage while the manager is active, None otherwise.
        """
        if self.core is None or not (self._gated or self.active):
            return None
        return (
            id(self),
            tuple((a.ptr, a.shape, a.strides, a.dtype) for a in self.core.owners),
        )

    @property
    def capture_owners(self):
        return () if self.core is None else self.core.owners

    def _issue(self, body):
        """Issue the launches of one hook: behind the device flag in a graph, by ``active`` otherwise.

        Inside a capture on a device with conditional graphs the launches are recorded
        behind ``core.enabled``, whatever ``active`` is now: a later launch of that graph
        follows the flag of its own attempt. Two other forms are excluded on purpose. A
        conditional outside a capture would read the flag back and wait for the stream;
        the host knows the decision there. Issuing the launches unconditionally would
        apply the basis of an inactive manager.
        """
        if self._gated and self.solver.model.device.is_capturing:
            wp.capture_if(self.core.enabled, body)
        elif self.active:
            body()

    def before_solve(self, physical_rhs, linear):
        if self.core is None or not getattr(linear, "compact_solve", False):
            return
        core, fixed, maximum = self.core, self.solver._fixed_particles, self.solver._rhs_maximum
        self._issue(lambda: core.before(physical_rhs, fixed.compact_rhs, fixed.compact_increment, maximum))

    def after_solve(self, physical_rhs, linear):
        if self.core is None or not getattr(linear, "compact_solve", False):
            return
        core, fixed, maximum = self.core, self.solver._fixed_particles, self.solver._rhs_maximum
        self._issue(lambda: core.after(physical_rhs, fixed.compact_increment, maximum, linear))

    def diagnostics(self):
        # Zeros from the moment the storage exists, which start-up makes ahead of the first track.
        counters = {} if self.core is None else dict(zip(COUNTERS, self.core.counters.numpy().tolist(), strict=True))
        return {
            "status": self.status,
            "basis_builds": self.basis_builds,
            "support_dofs": len(self.support),
            "refresh_generations": self.refresh_generations,
            "preparations": self.preparation_reports,
            "counters": counters,
        }
