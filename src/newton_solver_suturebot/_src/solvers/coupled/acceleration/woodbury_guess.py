# SPDX-License-Identifier: Apache-2.0
"""Guarded shaft-PC guess from an existing FEM response basis.

Host packet and small dense solve are outside capture; captured trial loops
compute the same coefficients with device kernels (``woodbury_device``). The
original current operator and unchanged PCG remain authoritative. No physical
state is owned.

One guess serves a correction for its lifetime. It binds the response core of
a FEM structure once (the free-node map, read once, and the storage that the
kernels address) and from then on has one key for a captured trial loop.
Whether the bound basis may be used by a launch is device data (``available``,
written by the host where it changes): a captured start is a conditional on
it, the device coefficients and their certified guess on one side, the zero
start on the other. The support of the basis and the number of track samples
are device data as well, so a new track sample changes no graph.

The anchor tangent of the small system is the one the operator applies: D, or
(I + S) D for a sliding anchor of a friction tunnel with its shear S (see
``shaft_preconditioner``). The system is then not symmetric; the host solve is
general and takes it as it is.
"""

import time

import numpy as np
import warp as wp

from ..interface.insertion import LIVE_TRACK

_BLOCK = 256
# Scratch of the start guess. A bite of 10 mm at the track spacing of 0.25 mm has 40 track
# samples; 48 samples (144 rows) is the largest tile the shared memory of the device holds
# (83 kB of 101 kB). A track sample lies in one tetrahedron, so it has at most four support
# nodes: 4 * 48 support columns.
_MAX_TRACKS = 48
_CAPACITY = 4 * _MAX_TRACKS
_HEADER = 2 + 36 + _CAPACITY + 6


@wp.kernel(enable_backward=False)
def gather_compliance(
    basis: wp.array[float],
    support: wp.array[int],
    count: int,
    stride: int,
    output: wp.array2d[float],
):
    row, column = wp.tid()
    output[row, column] = basis[column * stride + support[row]]


@wp.kernel(enable_backward=False)
def pack_head(
    rigid: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    free: wp.array[int],
    support: wp.array[int],
    support_count: wp.array[int],
    surface_count: wp.array[int],
    surface_valid: wp.array[int],
    packet: wp.array[float],
):
    packet[0] = float(surface_count[0])
    packet[1] = float(surface_valid[0])
    for row in range(6):
        for col in range(6):
            block = row // 3
            if col // 3 != row // 3:
                block += 2
            packet[2 + 6 * row + col] = rigid[block][row % 3, col % 3]
    for column in range(support_count[0]):
        dof = support[column]
        packet[38 + column] = rhs[free[dof // 3]][dof % 3]
    for axis in range(6):
        packet[38 + _CAPACITY + axis] = rhs[rhs.shape[0] - 2 + axis // 3][axis % 3]


@wp.kernel(enable_backward=False)
def pack_tracks(
    nodes: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    live: wp.array[int],
    matrices: wp.array[wp.mat33],
    levers: wp.array[wp.vec3],
    shears: wp.array[wp.mat33],
    packet: wp.array[float],
):
    i = wp.tid()
    if i >= live[LIVE_TRACK]:
        return
    begin = _HEADER + 20 * i
    for j in range(4):
        packet[begin + j] = float(nodes[i][j])
        packet[begin + 4 + j] = bary[i][j]
    # The whole anchor tangent (I + S) D; without a shear that is D itself.
    tangent = matrices[i] + shears[i] * matrices[i]
    for row in range(3):
        for col in range(3):
            packet[begin + 8 + 3 * row + col] = tangent[row, col]
    for j in range(3):
        packet[begin + 17 + j] = levers[i][j]


@wp.kernel(enable_backward=False)
def build_guess(
    enabled: wp.array[int],
    basis: wp.array[float],
    support_count: wp.array[int],
    stride: int,
    inverse_free: wp.array[int],
    coefficients: wp.array[float],
    rhs: wp.array[wp.vec3],
    guess: wp.array[wp.vec3],
):
    node = wp.tid()
    result = wp.vec3(0.0)
    if enabled[0] != 0:
        if node >= inverse_free.shape[0]:
            for axis in range(3):
                result[axis] = coefficients[_CAPACITY + 3 * (node - inverse_free.shape[0]) + axis]
        else:
            compact = inverse_free[node]
            if compact < 0:
                result = rhs[node]
            else:
                for column in range(support_count[0]):
                    base = column * stride + 3 * compact
                    result += coefficients[column] * wp.vec3(basis[base], basis[base + 1], basis[base + 2])
    guess[node] = result


@wp.func
def _squared_rhs(r: float, p: float):
    return wp.float64(r) * wp.float64(r)


@wp.func
def _squared_error(r: float, p: float):
    difference = wp.float64(r) - wp.float64(p)
    return difference * difference


@wp.func
def _nonfinite(r: float, p: float):
    result = wp.float64(0.0)
    if not wp.isfinite(r) or not wp.isfinite(p):
        result = wp.float64(1.0)
    return result


@wp.kernel(enable_backward=False)
def certificate_partials(rhs: wp.array[float], product: wp.array[float], partials: wp.array2d[wp.float64]):
    block, _lane = wp.tid()
    r = wp.tile_load(rhs, shape=_BLOCK, offset=block * _BLOCK)
    p = wp.tile_load(product, shape=_BLOCK, offset=block * _BLOCK)
    wp.tile_store(partials[0], wp.tile_sum(wp.tile_map(_squared_rhs, r, p)), offset=block)
    wp.tile_store(partials[1], wp.tile_sum(wp.tile_map(_squared_error, r, p)), offset=block)
    wp.tile_store(partials[2], wp.tile_sum(wp.tile_map(_nonfinite, r, p)), offset=block)


@wp.kernel(enable_backward=False)
def finish_certificate(
    partials: wp.array2d[wp.float64],
    enabled: wp.array[int],
    selected: wp.array[int],
    counters: wp.array[int],
):
    norm, error, bad = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    for block in range(partials.shape[1]):
        norm += partials[0, block]
        error += partials[1, block]
        bad += partials[2, block]
    selected[0] = int(
        enabled[0] != 0 and bad == wp.float64(0.0) and wp.isfinite(norm) and wp.isfinite(error) and error < norm
    )
    counters[0] += 1
    counters[1] += selected[0]


@wp.kernel(enable_backward=False)
def cpu_certificate(
    rhs: wp.array[float],
    product: wp.array[float],
    enabled: wp.array[int],
    selected: wp.array[int],
    counters: wp.array[int],
):
    norm, error, bad = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    for i in range(rhs.shape[0]):
        norm += _squared_rhs(rhs[i], product[i])
        error += _squared_error(rhs[i], product[i])
        bad += _nonfinite(rhs[i], product[i])
    selected[0] = int(
        enabled[0] != 0 and bad == wp.float64(0.0) and wp.isfinite(norm) and wp.isfinite(error) and error < norm
    )
    counters[0] += 1
    counters[1] += selected[0]


@wp.kernel(enable_backward=False)
def discard_guess(selected: wp.array[int], guess: wp.array[wp.vec3]):
    i = wp.tid()
    if selected[0] == 0:
        guess[i] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def certified_residual(
    selected: wp.array[int],
    rhs: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
):
    i = wp.tid()
    value = rhs[i]
    if selected[0] != 0:
        value -= product[i]
    residual[i] = value


@wp.kernel(enable_backward=False)
def collect_iterations(
    active: wp.array[int],
    iterations: wp.array[int],
    maximum: int,
    selected: wp.array[int],
    surface_active: wp.array[int],
    counters: wp.array[int],
):
    if active[0] == 0:
        return
    counters[2] += iterations[0]
    counters[3] += int(iterations[0] >= maximum)
    counters[4] += surface_active[0]
    counters[5] += surface_active[0] * selected[0]


class HostSchurPlan:
    """Cache immutable free-node mapping and track interpolation only."""

    def __init__(self, support, global_free, compliance):
        self.support = tuple(support)
        self.mapping = {int(dof): column for column, dof in enumerate(support)}
        self.inverse = {int(node): index for index, node in enumerate(global_free)}
        self.compliance = compliance
        self.topology = None
        self.p = self.pc = self.pcp = None

    def interpolation(self, records):
        topology = records[:, :8].tobytes()
        if topology != self.topology:
            p = np.zeros((3 * len(records), len(self.support)))
            for track, record in enumerate(records):
                for node, weight in zip(record[:4], record[4:8], strict=True):
                    compact = self.inverse.get(int(node))
                    if compact is None or weight == 0:
                        continue
                    for axis in range(3):
                        column = self.mapping.get(3 * compact + axis)
                        if column is None:
                            raise ValueError("Current shaft contact exceeds basis support")
                        p[3 * track + axis, column] += weight
            pc = p @ self.compliance
            pcp = pc @ p.T
            self.p, self.pc, self.pcp = p, pc, pcp
            self.topology = topology
        return self.p, self.pc, self.pcp


def host_coefficients(packet, track_count, support, global_free, compliance, h, *, plan=None):
    """Small general Schur solve; all outputs are numerical guesses only."""
    packet = np.asarray(packet, dtype=float)
    support = tuple(support)
    # Surface terms are deliberately omitted only from this approximate
    # proposal. The original combined operator certifies its true residual.
    if not 0 < track_count <= _MAX_TRACKS or not 0 < len(support) <= _CAPACITY:
        raise ValueError("Unsupported contact or support capacity")
    if not np.isfinite(packet[: _HEADER + 20 * track_count]).all():
        raise ValueError("Nonfinite contact packet")
    if not np.isfinite(h) or h <= 0 or compliance.shape != (len(support), len(support)):
        raise ValueError("Invalid basis compliance or timestep")
    if plan is None:
        plan = HostSchurPlan(support, global_free, compliance)
    elif plan.support != support or plan.compliance is not compliance:
        raise ValueError("Schur plan does not belong to this response basis")
    records = packet[_HEADER : _HEADER + 20 * track_count].reshape(track_count, 20)
    p, pc, pcp = plan.interpolation(records)
    g = np.zeros((3 * track_count, 6))
    d = np.zeros((3 * track_count, 3 * track_count))
    for track in range(track_count):
        record = packet[_HEADER + 20 * track : _HEADER + 20 * (track + 1)]
        rows = slice(3 * track, 3 * track + 3)
        x, y, z = record[17:20]
        g[rows, :3] = np.eye(3)
        g[rows, 3:] = -np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        d[rows, rows] = record[8:17].reshape(3, 3)
    mass = packet[2:38].reshape(6, 6)
    r_tissue = packet[38 : 38 + len(support)]
    r_rigid = packet[38 + _CAPACITY : _HEADER]
    rigid = np.linalg.solve(mass, np.column_stack((g.T, r_rigid)))
    c = pcp + g @ rigid[:, :-1]
    relative = -pc @ r_tissue + g @ rigid[:, -1]
    weights = np.linalg.solve(np.eye(len(d)) + h * d @ c, d @ relative)
    coefficients = np.zeros(_CAPACITY + 6, dtype=np.float32)
    coefficients[: len(support)] = r_tissue + h * p.T @ weights
    coefficients[_CAPACITY:] = rigid[:, -1] - h * rigid[:, :-1] @ weights
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite Woodbury coefficients")
    return coefficients


class WoodburyInitialGuess:
    """Host packet path: one readback per application, plus the compliance at binding changes.

    Captured trial loops (``prepare_device``, ``device_coefficients``): no readback but
    the free nodes of a newly bound FEM structure.
    """

    def __init__(self, preconditioner):
        self.pre = preconditioner
        device = self.pre.device
        self.packet = wp.zeros(_HEADER + 20 * _MAX_TRACKS, dtype=float, device=device)
        self.coefficients = wp.zeros(_CAPACITY + 6, dtype=float, device=device)
        self.enabled = wp.zeros(1, dtype=int, device=device)
        self.selected = wp.zeros(1, dtype=int, device=device)
        self.counters = wp.zeros(6, dtype=int, device=device)
        self.surface_active = wp.zeros(1, dtype=int, device=device)
        self.no_surface = wp.zeros(1, dtype=int, device=device)
        # Flag of ``collect_iterations``: this guess counts every solve it started.
        self.one = wp.ones(1, dtype=int, device=device)
        self.product = wp.zeros_like(self.pre.z)
        self.blocks = (3 * self.pre.n + _BLOCK - 1) // _BLOCK
        self.partials = wp.zeros((3, self.blocks), dtype=wp.float64, device=device)
        self.rhs_scalar = wp.array(ptr=self.pre.rhs.ptr, shape=(3 * self.pre.n,), dtype=float, device=device)
        self.product_scalar = wp.array(ptr=self.product.ptr, shape=(3 * self.pre.n,), dtype=float, device=device)
        # Device flag of a captured start: nonzero while the bound basis may be used. The host
        # writes it where it changes (``usable`` is its value on the host).
        self.available = wp.zeros(1, dtype=int, device=device)
        self.usable = False
        # True while a captured trial loop issues a correction: the start then takes its
        # coefficients from ``device_solve`` behind ``available``, not from a host packet.
        self.device_coefficients = False
        # Compact index of every tissue node, -1 for a fixed one: one storage for the
        # lifetime of this object, rewritten in place for a newly bound structure.
        self.inverse_free = wp.full(self.pre.n - 2, -1, dtype=int, device=device)
        # Keys of the bound FEM structure and of the host binding (compliance and plan).
        self.structure = None
        self.key = None
        self._packet_preparation = None
        self.device_solve = None
        self._product_current = False
        self.core = None
        self.preparations = self.ready = self.fallbacks = 0
        self.fallback_reasons = {}
        self.last_reason = "unbound"
        self.timings = dict.fromkeys(
            (
                "prepare_wall_seconds",
                "binding_seconds",
                "packet_enqueue_and_wait_seconds",
                "host_dense_seconds",
                "coefficient_upload_enqueue_seconds",
            ),
            0.0,
        )

    @property
    def owners(self):
        owned = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        if self.core is None:
            return owned
        # The kernels read the support count and the track count from the last two.
        live = self.pre.coupler.interaction._live
        return (*owned, self.core.basis, self.core.free, self.core.support, self.core.count, live)

    def _enqueue_packet(self, sc, sv):
        wp.launch(
            pack_head,
            1,
            [
                self.pre.rigid_blocks,
                self.pre.rhs,
                self.core.free,
                self.core.support,
                self.core.count,
                sc,
                sv,
                self.packet,
            ],
            device=self.pre.device,
        )
        interaction = self.pre.coupler.interaction
        wp.launch(
            pack_tracks,
            _MAX_TRACKS,
            [
                interaction._track_nodes,
                interaction._track_bary,
                interaction._live,
                self.pre.matrices,
                self.pre.levers,
                self.pre.shears,
                self.packet,
            ],
            device=self.pre.device,
        )

    def prepare(self, manager):
        start = time.perf_counter()
        try:
            return self._prepare(manager)
        finally:
            self.timings["prepare_wall_seconds"] += time.perf_counter() - start

    def _usable_basis(self, manager):
        """Whether the response basis of ``manager`` may start this correction; False selects the zero start."""
        if manager is None or not manager.active or manager.core is None:
            self.last_reason = "inactive FEM basis manager"
            return False
        if not 0 < len(manager.support) <= _CAPACITY or not 0 < self.pre.count <= _MAX_TRACKS:
            self.last_reason = "capacity fallback"
            return False
        if manager.structural_key != manager._structure():
            self.last_reason = "changed compact/free binding"
            return False
        return True

    def _bind_structure(self, manager):
        """Bind the response core of the current FEM structure; outside a capture.

        Once per structure: the free nodes are read and the node map is rewritten in place,
        so its storage is the one that every graph of this guess addresses.
        """
        core = manager.core
        structure = (
            id(manager),
            id(core),
            manager.structural_key,
            tuple((a.ptr, a.shape, a.strides, a.dtype) for a in (core.basis, core.free, core.support, core.count)),
        )
        if structure == self.structure:
            return
        started = time.perf_counter()
        self.core = core
        self.free_host = core.free.numpy()
        inverse = np.full(self.pre.n - 2, -1, dtype=np.int32)
        inverse[self.free_host] = np.arange(len(self.free_host))
        self.inverse_free.assign(inverse)
        self.structure = structure
        # Neither the compliance of the host path nor the column map of the device path is bound to it yet.
        self.key = None
        if self.device_solve is not None:
            self.device_solve.bind_structure()
        self.timings["binding_seconds"] += time.perf_counter() - started

    def _bind_host(self, manager):
        """Host packet path only: the compliance on the support of the basis, read once per basis build."""
        # Also where the binding is current: a captured launch in between follows the manager as well.
        self.support = tuple(manager.support)
        key = (self.structure, self.support, manager.basis_builds)
        if key == self.key:
            return
        started = time.perf_counter()
        small = wp.empty(
            (len(self.support), len(self.support)),
            dtype=float,
            device=self.pre.device,
        )
        wp.launch(
            gather_compliance,
            small.shape,
            [
                self.core.basis,
                self.core.support,
                len(self.support),
                3 * len(self.free_host),
                small,
            ],
            device=self.pre.device,
        )
        self.compliance = small.numpy().astype(float)
        self.host_plan = HostSchurPlan(self.support, self.free_host, self.compliance)
        self.key = key
        self.timings["binding_seconds"] += time.perf_counter() - started

    def _prepare(self, manager):
        self.surface_active.zero_()
        self.enabled.zero_()
        self.preparations += 1
        if not self._usable_basis(manager):
            return
        self._bind_structure(manager)
        self._bind_host(manager)
        started = time.perf_counter()
        surface = getattr(self.pre, "surface", None)
        sc = self.no_surface if surface is None else surface.anchors.count
        sv = self.no_surface if surface is None else surface.valid
        if self._packet_preparation is None:
            from .contact_preparation import PacketPreparation

            self._packet_preparation = PacketPreparation(self)
        self._packet_preparation.run(sc, sv)
        packet = self.packet.numpy()
        self.timings["packet_enqueue_and_wait_seconds"] += time.perf_counter() - started
        self.surface_active.fill_(int(packet[0] > 0 and packet[1] != 0))
        started = time.perf_counter()
        try:
            values = host_coefficients(
                packet,
                self.pre.count,
                self.support,
                self.free_host,
                self.compliance,
                float(np.float32(self.pre._last_dt)),
                plan=self.host_plan,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self.fallbacks += 1
            self.last_reason = str(error)
            self.fallback_reasons[self.last_reason] = self.fallback_reasons.get(self.last_reason, 0) + 1
            return
        finally:
            self.timings["host_dense_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        self.coefficients.assign(values)
        self.enabled.fill_(1)
        self.timings["coefficient_upload_enqueue_seconds"] += time.perf_counter() - started
        self.ready += 1
        self.last_reason = "current residual certification pending"

    def prepare_device(self, manager):
        """Host half of the captured guess: bind the basis, no packet and no dense solve.

        Call outside capture before every launch of a trial loop. Returns None while no
        response core was ever bound, so the zero start must be captured instead. A core
        is bound as soon as the manager holds one for the current FEM structure, with or
        without a basis in it: a manager whose storage is made ahead of the first track
        support (``AffineGuessManager.prepare``) therefore gives the key of the run at its
        first call. From the first binding on this returns the key of everything a
        captured start bakes into a graph, which changes with the FEM structure only:
        whether this launch may use the basis (an active manager of an unchanged
        structure, tracks and support within the fixed scratch) is written to
        ``available``, the padded tile of the track count to the size flags of
        ``device_solve``, and the support of the basis to its column map, each where it
        changed.
        """
        usable = self._usable_basis(manager)
        if usable or (
            manager is not None
            and manager.core is not None
            and manager.structural_key is not None
            and manager.structural_key == manager._structure()
        ):
            # The storage that a captured start addresses: the core, the node map and the
            # scratch of the device solve. None of it depends on the support of the basis.
            self._bind_structure(manager)
            if self.device_solve is None:
                from .woodbury_device import DeviceSchurCoefficients

                self.device_solve = DeviceSchurCoefficients(self)
                self.device_solve.bind_structure()
        if usable:
            # The host packet path binds the same guess between captured launches: both
            # follow the support of the manager, each with its own binding.
            self.support = tuple(manager.support)
            self.device_solve.bind_support(self.support)
            self.device_solve.select(self.pre.count)
        if usable != self.usable:
            self.available.fill_(int(usable))
            self.usable = usable
        if self.device_solve is None:
            return None
        core = self.core
        return (
            id(self),
            id(core),
            3 * len(self.free_host),
            tuple(
                (a.ptr, a.shape, a.dtype)
                for a in (
                    core.basis,
                    core.free,
                    core.support,
                    core.count,
                    self.inverse_free,
                    self.rhs_scalar,
                    self.available,
                    self.pre.coupler.interaction._live,
                )
            ),
            self.device_solve.key(),
            # The launches of a friction tunnel hold the normal equations.
            bool(self.pre.nonsymmetric),
        )

    def initialize(self):
        if self.device_coefficients:
            # A correction of a captured trial loop. The graph holds both starts and takes
            # one by the flag of each launch; outside a capture the host knows the flag.
            if self.pre.device.is_capturing:
                wp.capture_if(self.available, on_true=self._device_start, on_false=self._zero_start)
            elif self.usable:
                self._device_start()
            else:
                self._zero_start()
            # Either start serves the first residual of the solve: see ``first_residual``.
            self._product_current = True
            return
        if self.core is None:
            self.pre.z.zero_()
            return
        self._certified_start()

    def _device_start(self):
        """Coefficients of this correction from the device kernels, then their certified guess."""
        self.device_solve.enqueue()
        self._certified_start()

    def _zero_start(self):
        """No usable basis: the zero vector, marked as a discarded guess. No scratch of the guess is touched."""
        self.pre.z.zero_()
        self.selected.zero_()

    def _certified_start(self):
        """The guess of the current coefficients, kept only if the current operator certifies it."""
        wp.launch(
            build_guess,
            self.pre.n,
            [
                self.enabled,
                self.core.basis,
                self.core.count,
                3 * len(self.free_host),
                self.inverse_free,
                self.coefficients,
                self.pre.rhs,
                self.pre.z,
            ],
            device=self.pre.device,
        )
        self.pre.operator.matvec(self.pre.z, self.product, self.product, 1.0, 0.0)
        if self.pre.device.is_cuda:
            wp.launch(
                certificate_partials,
                (self.blocks, _BLOCK),
                [self.rhs_scalar, self.product_scalar, self.partials],
                block_dim=_BLOCK,
                device=self.pre.device,
            )
            wp.launch(
                finish_certificate,
                1,
                [self.partials, self.enabled, self.selected, self.counters],
                device=self.pre.device,
            )
        else:
            wp.launch(
                cpu_certificate,
                1,
                [
                    self.rhs_scalar,
                    self.product_scalar,
                    self.enabled,
                    self.selected,
                    self.counters,
                ],
                device=self.pre.device,
            )
        wp.launch(
            discard_guess,
            self.pre.n,
            [self.selected, self.pre.z],
            device=self.pre.device,
        )
        self._product_current = True

    def first_residual(self, x, y, z, alpha, beta):
        """Serve b - A x of the certified start from the product the certificate took.

        The certificate applied the current operator to this same vector. A
        selected guess leaves b minus that product, a discarded one is zero
        and leaves b: the values a second operator application would return.
        The zero start of an unavailable basis is such a discarded guess.
        """
        if not self._product_current or x is not self.pre.z or alpha != -1.0 or beta != 1.0:
            return False
        self._product_current = False
        wp.launch(certified_residual, self.pre.n, [self.selected, y, self.product, z], device=self.pre.device)
        return True

    def after_solve(self):
        self._product_current = False
        linear = self.pre.linear_solver
        wp.launch(
            collect_iterations,
            1,
            [
                # A captured start counts the solves of an available basis only.
                self.available if self.device_coefficients else self.one,
                linear._cur_iter_and_condition,
                linear._maxiter,
                self.selected,
                self.surface_active,
                self.counters,
            ],
            device=self.pre.device,
        )

    def diagnostics(self):
        # Call with ordinary end-of-run diagnostics, never on each trial.
        return {
            "preparations": self.preparations,
            "host_ready": self.ready,
            "host_fallbacks": self.fallbacks,
            "fallback_reasons": dict(self.fallback_reasons),
            "last_reason": self.last_reason,
            "timings": dict(self.timings),
            "counters": dict(
                zip(
                    (
                        "applications",
                        "selected",
                        "cg_iterations",
                        "cg_cap_hits",
                        "surface_applications",
                        "surface_selected",
                    ),
                    self.counters.numpy().tolist(),
                    strict=True,
                )
            ),
        }
