# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The cut simulation as one object: a tetrahedral body on a floor that is cut, gripped and stepped.

:class:`CutTissue` joins the parts of this folder into the simulation that a cutting scene runs:
the pool of node copies and the cutter (:class:`LinearTetCut`), the cut models
(:class:`CutModel`), the boundary surface and its contact (:class:`CutSurfaceContact`), the sets
of grips (:class:`MaterialHolds`), the volume guard (:class:`CutVolumeGuard`) and the FEM
solver. It owns the frame and its CUDA graph, the start-up that builds and captures all of it,
the transaction of an edit with its roll-back, the growth of storage and the fixed reserve.

A scene supplies the body, its numbers and the user: it loads the mesh, places the edits and
the grips, and draws. Nothing here knows an asset, a screen tool or a viewer.
"""

import gc
import math
from time import perf_counter

import newton
import numpy as np
import warp as wp

from ..solvers.fem import SolverFEM
from ..utils import capture
from .contact import CutSurfaceContact
from .holds import MaterialHolds
from .model import CutModel
from .tetrahedral_cut import LinearTetCut
from .volume_guard import CutVolumeGuard


@wp.kernel
def _clone_nodes(
    sources: wp.array2d[int],
    count: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    copied_q: wp.array[wp.vec3],
    copied_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < count[0]:
        target, source = sources[i, 0], sources[i, 1]
        copied_q[target] = q[source]
        copied_qd[target] = qd[source]


@wp.kernel(enable_backward=False)
def _node_components(
    elements: wp.array2d[int],
    fractions: wp.array[float],
    components: wp.array[int],
    node_components: wp.array[int],
):
    element = wp.tid()
    if fractions[element] > 0.0:
        for corner in range(4):
            wp.atomic_max(node_components, elements[element, corner], components[element])


# What the fixed reserve is derived from (CutTissue.derive_reserve), beside the radius of the
# scene's tool: the planes of a cylinder (16 sides, two ends) and of a swept sphere (20 faces of
# the icosahedron and at most 10 sides of its sweep), the triangles of a section through a
# tetrahedron (a quadrilateral on each lip) and of the wall that one tool plane leaves in a
# tetrahedron (a triangular tool face cut by four faces has at most seven corners).
CYLINDER_PLANES, SPHERE_PLANES = 18, 30
SECTION_TRIANGLES, WALL_TRIANGLES = 4, 5
# Polygon vertices and faces that the reserve holds for every fragment: the sizes that the cutter's own
# growth reaches first (from 12 and 12). They are not a bound. A tetrahedron that every plane of a swept
# sphere crosses can leave pieces of 4 + 2 * 30 = 64 vertices and 4 + 30 = 34 faces; the cutter's limits
# are 64 and 48. The polygon storage is most of the reserve, so a larger one is asked for by name.
RESERVE_POLYGONS = (48, 32)


def _plain_rejection_message(error, fragment_capacity, *, fixed=False):
    """The text of a refused edit for a scene that gives none of its own."""
    return f"Cut rejected: {error}"


class CutTissue:
    """A tetrahedral body on a floor: cut, gripped by springs and stepped, with one frame graph.

    The constructor makes what no start-up replaces: the pool of node copies as a particle-only
    model with the floor as its ground plane (``pool_model``; the cut models are copies of it,
    and a scene binds its viewer to it) and the factory of the cut models. :meth:`reset` is
    the start-up. The owner calls it once after construction and again for every Reset; until
    then there is nothing to step.

    Start-up (:meth:`reset`). The cutter with every graph of its edits captured and launched,
    three cut models (the live one, the staged one of an edit, the backup of a roll-back), the
    reserved boundary surface, its contact, the sets of grips anew, the volume guard and the
    solver; then, on a CUDA device, the frame is run once, put back, captured and launched once
    (:meth:`_capture_physics_graph`, :meth:`_launch_frame_graph`), so that no frame of the run
    captures a graph or builds its executable. :meth:`load_kernel_modules` loads the kernel
    module of this file. The marks of a start-up and what a viewer draws are the scene's.

    Frame. :meth:`step` counts a frame and advances it: the captured graph on CUDA, the same
    launches eagerly on the CPU (:meth:`simulate`). A scene whose user acts inside the frame
    calls :meth:`begin_frame`, applies its edits and calls :meth:`advance`; storage that grows
    in an edit is listed with the frame that was counted last. :meth:`_frame_snapshot` keeps
    what a frame changes, for a frame that runs outside the run.

    Edits. :meth:`cut`, :meth:`cut_sphere` and :meth:`cut_cylinder` each propose an edit to the
    cutter and publish it as one transaction: cloned nodes, the staged model, the grips' owners,
    the solver's topology and the contact's surface. An edit that fails anywhere is rolled back
    to the model, state, velocity increment, grips, surface and graph that were accepted;
    ``cut_error`` then holds its text. An edit beyond the reserved boundary triangles replaces
    the surface, the contact (which takes over the friction anchors) and the frame graph inside
    the edit; every growth is listed in ``storage_growths`` with its frame, sizes, seconds and
    captures. With ``reserve`` nothing grows: such an edit is refused.

    Grips. ``grips`` holds one mapping of :class:`MaterialHolds` keyword arguments for every
    set; :meth:`reset` makes the sets anew (``grip_sets``). Every set acts in the frame, reads
    the solver's step time, follows an edit and is put back by a refused one.

    A scene reads, beside the public attributes below, ``_surface_count`` (active boundary
    triangles) and ``_render_indices`` (the corner indices of the reserved boundary triangles,
    three unshared corners each, for drawing and picking ``contact.points``).

    Args:
        positions: Reference positions of the body [m], shape (N, 3).
        elements: Its tetrahedra, shape (M, 4).
        frame_dt: Time of one frame [s].
        substeps: Solver steps in a frame.
        solver_iterations: Conjugate-gradient iterations of a solver step.
        density: Material density [kg/m^3].
        young_modulus: Elastic modulus [Pa].
        poisson_ratio: Poisson ratio, in (0, 0.5).
        material_damping: Viscous material damping [Pa s].
        linear_damping: Mass-proportional damping [1/s].
        min_stretch: Lower stretch limit of the solver.
        max_stretch: Upper stretch limit of the solver.
        gravity: Gravity magnitude [m/s^2], along -z.
        floor_height: Height of the floor [m].
        ground_stiffness: Floor penalty [N/m^3].
        piece_stiffness: Penalty between cut pieces [N/m^3].
        search_radius: Reach of the contact search between pieces and of the friction carry [m].
        ground_friction: Coulomb friction coefficient of the floor; zero is a floor without friction.
        grips: One mapping of :class:`MaterialHolds` keyword arguments per set of grips.
        node_copies: Copies of every node in the pool.
        drop_height: Clearance of the start state above the reference positions [m].
        streaming_topology: Update cut matrix entries in persistent storage instead of rebuilding
            the sparse structure.
        fragment_capacity: Reserved material fragments; zero selects automatically.
        surface_capacity: Initial boundary triangle reserve; zero selects automatically.
        reserve: Fixed storage as :meth:`derive_reserve` returns it, or None: storage grows
            inside an edit.
        rejection_message: ``(error, fragment_capacity, fixed=...)`` to the text that
            ``cut_error`` holds after a refused edit; a scene names its own options there.
        device: Device of the simulation.

    Attributes:
        pool_model: Particle-only model of the pool of node copies, with the floor.
        model, state_0, state_1, control, solver: The live cut model, its two states, its
            control and its FEM solver.
        cutter: The cutter of the current start-up.
        surface, contact: The reserved boundary surface and its contact.
        grip_sets: The sets of grips of the current start-up, in the order of ``grips``.
        volume_guard: The volume guard.
        graph: The frame graph, or None on the CPU.
        fragment_count, cut_revision, cut_error: Active fragments, accepted edits since the
            start-up, and the text of the last refused edit (empty after an accepted one).
        last_cut_ms, last_geometry_ms, last_rebuild_ms: Host time of the last accepted edit, of
            its geometry and of its publication [ms].
        storage_growths: Every growth of storage inside an edit, with its frame.
        reserve: The fixed storage, or None.
        sim_time, sim_dt, frame_dt: Simulated time since the start-up, the time step of a
            solver step and the time of a frame [s].
        reference_positions, reference_elements: The body as it was given.
    """

    def __init__(
        self,
        positions,
        elements,
        *,
        frame_dt,
        substeps,
        solver_iterations,
        density,
        young_modulus,
        poisson_ratio,
        material_damping,
        linear_damping,
        min_stretch,
        max_stretch,
        gravity,
        floor_height,
        ground_stiffness,
        piece_stiffness,
        search_radius,
        ground_friction=0.0,
        grips=(),
        node_copies=4,
        drop_height=0.0,
        streaming_topology=True,
        fragment_capacity=0,
        surface_capacity=0,
        reserve=None,
        rejection_message=None,
        device=None,
    ):
        self.device = wp.get_device(device)
        self.frame_dt, self.substeps, self.solver_iterations = frame_dt, substeps, solver_iterations
        self.linear_damping, self.min_stretch, self.max_stretch = linear_damping, min_stretch, max_stretch
        self.floor_height, self.ground_stiffness, self.ground_friction = floor_height, ground_stiffness, ground_friction
        self.piece_stiffness, self.search_radius = piece_stiffness, search_radius
        self.node_copies, self.drop_height, self.streaming_topology = node_copies, drop_height, streaming_topology
        self.fragment_capacity, self.surface_capacity = fragment_capacity, surface_capacity
        self._grip_parameters = tuple(dict(parameters) for parameters in grips)
        self._rejection_message = _plain_rejection_message if rejection_message is None else rejection_message
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / substeps
        self.reference_positions, self.reference_elements = positions, elements
        self._pool_size = len(self.reference_positions) * node_copies
        builder = newton.ModelBuilder()
        pool = np.tile(self.reference_positions, (node_copies, 1))
        builder.add_particles(
            pool.tolist(), np.zeros_like(pool).tolist(), np.ones(len(pool)).tolist(), radius=[0.0] * len(pool)
        )
        builder.add_ground_plane(height=self.floor_height)
        self.pool_model = builder.finalize(device=self.device)
        self.pool_model.set_gravity((0.0, 0.0, -gravity))
        self._model_factory = CutModel(
            self.pool_model,
            self.reference_positions,
            self.reference_elements,
            density=density,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            damping=material_damping,
        )
        # Frames stepped so far, less one; and every growth of storage inside an edit, with its frame.
        self._frame = -1
        self.storage_growths = []
        # Fixed storage (derive_reserve), or None: storage grows inside an edit.
        self.reserve = reserve
        self.cutter = None

    @staticmethod
    def derive_reserve(positions, elements, scale, polygons=None, *, tool_radius):
        """Fixed storage for ``scale`` design loads of cutting, derived from the mesh and the tools.

        One design load is a line cut through the whole depth across the body in each horizontal
        direction, one cylinder and one sphere tunnel through the whole depth. The counts are taken on
        this mesh: the tetrahedra that the two sections through its centre cross, and the tetrahedra
        whose bounding box a vertical tool of ``tool_radius`` can touch at the fullest place
        of the body (the cutter selects the fragments of an edit by their bounding boxes).

        - fragments: one more for every tetrahedron a section crosses (its second half), and one less
          than the planes of the tool for every tetrahedron of a tool column (a convex tool leaves at
          most one piece per plane).
        - subtractor work: the tetrahedra of a tool column.
        - boundary triangles, beyond those of the uncut body: ``SECTION_TRIANGLES`` per crossed
          tetrahedron, ``WALL_TRIANGLES`` per tool plane and tetrahedron of a tool column.
        - polygon vertices and faces per fragment: ``RESERVE_POLYGONS``, or ``polygons``. This part is
          not derived from the mesh; an edit that needs more is refused.
        - subtractor scratch: the cutter's limits.

        Args:
            positions: Reference positions of the mesh [m].
            elements: Its tetrahedra.
            scale: Design loads; a fraction reserves less.
            polygons: Polygon vertices and faces per fragment, or None for ``RESERVE_POLYGONS``.
            tool_radius: Radius of the scene's cylinder or sphere tool [m].

        Returns:
            The reserve as a dict of capacities, with the counts they were derived from.
        """
        corners = positions[elements]
        lower, upper = corners.min(axis=1), corners.max(axis=1)
        low, high = positions.min(axis=0), positions.max(axis=0)
        center = 0.5 * (low + high)
        crossed = sum(
            int(np.count_nonzero((lower[:, axis] < center[axis]) & (upper[:, axis] > center[axis]))) for axis in (0, 1)
        )
        # Tool centres on a grid of one radius: a tool whose centre lies in a cell touches only boxes
        # within a radius and half a cell of the cell's centre. Counted with a two-dimensional prefix sum.
        spacing, margin = tool_radius, 1.5 * tool_radius
        shape = np.ceil((high[:2] - low[:2]) / spacing).astype(int) + 2
        first = np.clip(np.ceil((lower[:, :2] - margin - low[:2]) / spacing).astype(int), 0, shape - 1)
        last = np.clip(np.floor((upper[:, :2] + margin - low[:2]) / spacing).astype(int), 0, shape - 1)
        grid = np.zeros(shape + 1, dtype=np.int64)
        np.add.at(grid, (first[:, 0], first[:, 1]), 1)
        np.add.at(grid, (last[:, 0] + 1, first[:, 1]), -1)
        np.add.at(grid, (first[:, 0], last[:, 1] + 1), -1)
        np.add.at(grid, (last[:, 0] + 1, last[:, 1] + 1), 1)
        column = int(grid.cumsum(axis=0).cumsum(axis=1).max())
        tool_planes = CYLINDER_PLANES + SPHERE_PLANES
        if polygons is None:
            polygons = RESERVE_POLYGONS
        return {
            "scale": float(scale),
            "crossed_tetrahedra": crossed,
            "column_tetrahedra": column,
            "fragment_capacity": len(elements) + math.ceil(scale * (crossed + (tool_planes - 2) * column)),
            "work_capacity": max(256, math.ceil(scale * column)),
            "surface_triangles": math.ceil(
                scale * (SECTION_TRIANGLES * crossed + WALL_TRIANGLES * tool_planes * column)
            ),
            "vertex_capacity": int(polygons[0]),
            "face_capacity": int(polygons[1]),
            "scratch_vertex_capacity": LinearTetCut.SCRATCH_LIMIT,
            "scratch_face_capacity": LinearTetCut.SCRATCH_LIMIT,
        }

    def load_kernel_modules(self):
        """Load the kernel module of this file, for a start-up.

        Its first launch that no start-up issues is the transfer of cloned nodes at the first
        accepted edit. The cutter loads its modules when it is made.
        """
        wp.load_module(_clone_nodes.module, device=self.device, block_dim=256 if self.device.is_cuda else 1)

    def _make_solver(self, model):
        solver = SolverFEM(
            model,
            tet_volume_weights=model.cut_volume_fractions,
            dt=self.sim_dt,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=self.solver_iterations,
            ground_plane=None,
            linear_damping=self.linear_damping,
            min_stretch=self.min_stretch,
            max_stretch=self.max_stretch,
        )
        if self.streaming_topology:
            solver.prepare_tetrahedral_updates(len(self.reference_positions))
        return solver

    def _install_physics(self, model, state, surface):
        self.model, self.state_0, self.state_1 = model, state, model.state()
        self.state_1.assign(state)
        self.control = model.control()
        self.solver = self._make_solver(model)
        # The grips, the contact and the volume guard act in the solver's step and read its
        # time step from the same device arrays as the solver.
        for grips in self.grip_sets:
            grips.step_time = self.solver.step_time
        self._backup_velocity_increment = wp.empty_like(self.solver.velocity_increment)
        self._mechanical_components = wp.clone(self.cutter.material_components)
        self._node_components = wp.empty(model.particle_count, dtype=int, device=self.device)
        self._update_mechanical_components(self.cutter.material_components)
        self.solver.prepare_component_scaling(self._node_components, model.tet_count)
        self.volume_guard = CutVolumeGuard(
            model, components=self._mechanical_components, step_time=self.solver.step_time
        )
        self.contact = CutSurfaceContact(
            surface,
            state.particle_q,
            model.particle_inv_mass,
            ground_height=self.floor_height,
            ground_stiffness=self.ground_stiffness,
            piece_stiffness=self.piece_stiffness,
            search_radius=self.search_radius,
            step_time=self.solver.step_time,
            ground_friction=self.ground_friction,
        )
        self.surface = surface
        self._surface_count = int(surface.count.numpy()[0])
        # A drawing list is uploaded once. Reserved triangles stay in it;
        # inactive point triples collapse to zero until a cut activates them.
        self._render_indices = wp.array(np.arange(surface.indices.size, dtype=np.int32), device=self.device)
        for grips in self.grip_sets:
            grips.remap(self.cutter.geometry)
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        self.graph = self._capture_physics_graph()
        self._launch_frame_graph()

    def _update_mechanical_components(self, components):
        # A growing surface warms the new model before the geometry transaction
        # commits. Its component labels must already match those new elements.
        wp.copy(self._mechanical_components, components)
        self._node_components.fill_(-1)
        wp.launch(
            _node_components,
            dim=self.model.tet_count,
            inputs=[
                self.model.tet_indices,
                self.model.cut_volume_fractions,
                self._mechanical_components,
                self._node_components,
            ],
            device=self.device,
        )

    def _frame_snapshot(self):
        """Keep what a frame changes; the returned call puts it back.

        The two states, the solver's velocity increment, the driven grip targets of every set,
        the count of the volume guard and the friction anchors of the floor: a frame that runs
        outside the run, eager or as a graph, leaves the scene as it found it.
        """
        states = (self.state_0, self.state_1)
        saved_states = tuple(self.model.state() for _ in states)
        for saved, state in zip(saved_states, states, strict=True):
            saved.assign(state)
        persistent = (
            self.solver.velocity_increment,
            *(grips.driven for grips in self.grip_sets),
            self.volume_guard.limited_steps,
            *self.contact.state,
        )
        saved_persistent = tuple(wp.clone(array) for array in persistent)

        def restore():
            for state, saved in zip(states, saved_states, strict=True):
                state.assign(saved)
            for array, saved in zip(persistent, saved_persistent, strict=True):
                wp.copy(array, saved)

        return restore

    def _capture_physics_graph(self):
        """Warm and capture the current buffers without advancing physical state."""
        if not self.device.is_cuda:
            return None
        restore = self._frame_snapshot()
        try:
            self.simulate()
            restore()
            with capture.ScopedCapture(device=self.device) as scope:
                self.simulate()
            return scope.graph
        finally:
            restore()

    def _launch_frame_graph(self):
        """Launch the captured frame once and put the scene back.

        Warp builds the executable of a graph at its first launch: that launch belongs to the
        start-up, or to the growth of the surface that captured the graph, not to the next frame.
        """
        if self.graph is None:
            return
        restore = self._frame_snapshot()
        try:
            wp.capture_launch(self.graph)
            wp.synchronize_device(self.device)
        finally:
            restore()

    def _list_growths(self, growths=()):
        """Add growths of storage to the list of this object, each with the frame it fell into.

        The growths that the cutter has listed since the last call come first: they precede a
        growth of the surface in the same edit.
        """
        cutter = self.cutter.storage_growths
        for growth in (*cutter[self._cutter_growths_listed :], *growths):
            self.storage_growths.append({"frame": self._frame, **growth})
        self._cutter_growths_listed = len(cutter)

    def _grow_surface(self, triangle_count):
        """Replace contact and drawing storage only when the boundary outgrows it."""
        self._list_growths()
        start = perf_counter()
        captured = capture.census.captures, capture.census.solver_captures
        growth = {"what": "boundary surface triangles", "from": self.contact.capacity, "completed": False}
        try:
            self._replace_surface(triangle_count)
            growth["completed"] = True
        finally:
            if self.device.is_cuda:
                wp.synchronize_device(self.device)
            growth["to"] = self.contact.capacity
            growth["seconds"] = perf_counter() - start
            growth["captures"] = capture.census.captures - captured[0]
            growth["solver_captures"] = capture.census.solver_captures - captured[1]
            self._list_growths([growth])

    def _replace_surface(self, triangle_count):
        capacity = max(1, self.contact.capacity)
        while capacity < triangle_count:
            capacity *= 2
        capacity = min(capacity, self.cutter.surface.components.size)
        surface = CutSurfaceContact.reserve_surface(self.cutter.surface, triangle_count, capacity)
        previous = self.contact
        contact = CutSurfaceContact(
            surface,
            self.state_0.particle_q,
            self.model.particle_inv_mass,
            ground_height=previous.ground_height,
            ground_stiffness=previous.ground_stiffness,
            piece_stiffness=previous.piece_stiffness,
            damping_ratio=previous.damping_ratio,
            search_radius=previous.search_radius,
            step_time=previous.step_time,
            ground_friction=previous.ground_friction,
        )
        contact.take_over(previous, self.state_0.particle_q)
        indices = wp.array(np.arange(surface.indices.size, dtype=np.int32), device=self.device)
        self.contact, self.surface, self._render_indices = contact, surface, indices
        # The FEM model, states, solver and sparse storage retain their addresses.
        # Only the graph's references to surface/contact buffers need replacing.
        self.graph = self._capture_physics_graph()
        self._launch_frame_graph()

    def simulate(self):
        """The launches of one frame: tangent refresh, the grips' matrix, then every substep."""
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        for grips in self.grip_sets:
            grips.add_matrix(self.solver, self.sim_dt)
        for substep in range(self.substeps):
            current, following = (self.state_0, self.state_1) if substep % 2 == 0 else (self.state_1, self.state_0)
            forces = self.contact.evaluate(current.particle_q, current.particle_qd, self.sim_dt, piece_contact=True)
            for grips in self.grip_sets:
                grips.add_forces(current.particle_q, current.particle_qd, forces, self.sim_dt)
            # The contact rows of this substep are in the solver's matrix for its solve.
            self.contact.add_matrix(self.solver)
            self.solver.step(current, following, self.control, None, self.sim_dt, external_forces=forces)
            self.contact.remove_matrix(self.solver)
            self.volume_guard.apply(
                following.particle_q, following.particle_qd, self.sim_dt, previous_q=current.particle_q
            )
        if self.substeps % 2:
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)

    def begin_frame(self):
        """Count a frame: storage that grows in an edit from here on is listed with it."""
        self._frame += 1

    def advance(self):
        """The physics of the counted frame: its graph on CUDA, the same launches eagerly on the CPU."""
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def step(self):
        """One frame in which no user acts: counted, then advanced."""
        self.begin_frame()
        self.advance()

    def reset(self):
        """Start up: the cutter, the models, the surface, the contact, the grips, the guard and the solver anew.

        Every graph is captured and launched here: those of the cutter's edits, for every patch
        count of a line cut and for the subtractor, and the frame graph. With ``reserve`` the
        storage of the edits is allocated here at its fixed size.
        """
        self.grip_sets = tuple(MaterialHolds(self.device, **parameters) for parameters in self._grip_parameters)
        self.cut_revision = 0
        source_count = len(self.reference_elements)
        fragment_capacity = self.fragment_capacity or source_count + 16384
        # The subtractor's scratch starts at the cutter's limit. From its default (48 vertices and the
        # polygon faces of the geometry, 12) the first cylinder grew it, 12 faces to 24, and captured
        # the subtractor's graph again in that frame. It is storage of the work list (256 entries),
        # not of the mesh: 0.58 GiB more than the default, on any body.
        polygons = (12, 12)
        subtractor = {
            "scratch_vertex_capacity": LinearTetCut.SCRATCH_LIMIT,
            "scratch_face_capacity": LinearTetCut.SCRATCH_LIMIT,
        }
        if self.reserve is not None:
            reserve = self.reserve
            fragment_capacity = self.fragment_capacity or reserve["fragment_capacity"]
            polygons = reserve["vertex_capacity"], reserve["face_capacity"]
            subtractor = {
                "work_capacity": reserve["work_capacity"],
                "scratch_vertex_capacity": reserve["scratch_vertex_capacity"],
                "scratch_face_capacity": reserve["scratch_face_capacity"],
            }
            if self.cutter is not None:
                # The reserve is as large as the storage it replaces: give that back first.
                self.cutter = None
                gc.collect()
        self.cutter = LinearTetCut(
            self.reference_positions,
            self.reference_elements,
            fragment_capacity=fragment_capacity,
            node_copies=self.node_copies,
            vertex_capacity=polygons[0],
            face_capacity=polygons[1],
            fixed_storage=self.reserve is not None,
            device=self.device,
        )
        self._cutter_growths_listed = 0
        # Every graph of the cutter and its subtractor: captured and launched here, not in the first
        # line cut of a patch count or in the first cylinder or sphere.
        self.cutter.prepare(patch_counts=tuple(range(1, self.cutter.max_patches + 1)))
        self.cutter.prepare_sphere(**subtractor)
        pool = np.tile(self.reference_positions, (self.node_copies, 1))
        pool[:, 2] += self.drop_height
        q = wp.array(pool, dtype=wp.vec3, device=self.device)
        qd = wp.zeros_like(q)
        self._cut_positions = wp.empty_like(q)
        self.fragment_count = len(self.reference_elements)
        model, state = self._build_model(self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd)
        self._staged_model, self._staged_state = self._build_model(
            self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd
        )
        self._backup_model, self._backup_state = self._build_model(
            self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd
        )
        surface_count = int(self.cutter.surface.count.numpy()[0])
        surface_capacity = max(2 * surface_count, surface_count + 256)
        if self.reserve is not None:
            surface_capacity = min(
                surface_count + self.reserve["surface_triangles"], self.cutter.surface.components.size
            )
        surface = CutSurfaceContact.reserve_surface(
            self.cutter.surface, surface_count, self.surface_capacity or surface_capacity
        )
        self._install_physics(model, state, surface)
        self.sim_time, self.cut_revision = 0.0, 0
        self.last_cut_ms, self.last_geometry_ms, self.last_rebuild_ms = 0.0, 0.0, 0.0
        self.cut_error = ""

    def _build_model(self, geometry, nodes, q, qd):
        return self._model_factory.build(
            geometry.parent,
            nodes,
            self.cutter.volume_fractions,
            self.cutter.mass_integrals,
            self.fragment_count,
            q,
            qd,
            capacity=self.cutter.capacity,
        )

    def cut(self, triangles):
        """Apply a finite incision given as world triangles [m]; leave a rejected edit unpublished."""
        return self._apply_edit(lambda positions: self.cutter.propose_triangles(positions, triangles))

    def cut_sphere(self, start, end, radius):
        """Remove the spherical tool's swept material; distances are in metres."""
        return self._apply_edit(lambda positions: self.cutter.propose_sphere(positions, start, end, radius))

    def cut_cylinder(self, start, end, radius):
        """Remove one straight, constant-radius cylinder; distances are in metres."""
        return self._apply_edit(lambda positions: self.cutter.propose_cylinder(positions, start, end, radius))

    def _apply_edit(self, propose):
        """Apply one edit; list the storage that grew inside it."""
        try:
            return self._publish_edit(propose)
        finally:
            self._list_growths()

    def _publish_edit(self, propose):
        start = perf_counter()
        published = False
        # The sets of grips whose owners were staged for the edit: a refused edit puts those back.
        staged_grips = 0
        previous_count, previous_surface_count = self.fragment_count, self._surface_count
        previous_surface_buffers = self.contact, self.surface, self._render_indices, self.graph
        try:
            wp.copy(self._cut_positions, self.state_0.particle_q)
            diagnostics = propose(self._cut_positions)
            if not diagnostics["changed_parents"]:
                self.cutter.discard()
                self.cut_error = ""
                return False
            surface_count = diagnostics["triangle_count"]
            self.last_geometry_ms = 1000.0 * (perf_counter() - start)
            self._staged_state.assign(self.state_0)
            wp.launch(
                _clone_nodes,
                dim=self._pool_size,
                inputs=[
                    self.cutter.candidate.clone_sources,
                    self.cutter.candidate.clone_count,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self._staged_state.particle_q,
                    self._staged_state.particle_qd,
                ],
                device=self.device,
            )
            self._model_factory.update(
                self._staged_model,
                self.cutter.pending_geometry.parent,
                self.cutter.candidate.nodes,
                self.cutter.volume_fractions,
                self.cutter.mass_integrals,
                diagnostics["fragment_count"],
            )
            for grips in self.grip_sets:
                staged_grips += 1
                grips.stage_remap(self.cutter.pending_geometry, self.cutter.candidate.nodes, self.cutter.geometry)
            self._model_factory.copy_into(self._backup_model, self.model)
            self._backup_state.assign(self.state_0)
            wp.copy(self._backup_velocity_increment, self.solver.velocity_increment)
            published = True
            self._model_factory.copy_into(self.model, self._staged_model)
            self.state_0.assign(self._staged_state)
            self.state_1.assign(self._staged_state)
            self._update_mechanical_components(self.cutter.candidate.material_components)
            self.solver.refresh_component_scaling()
            self.solver.refresh_topology(
                self.state_0.particle_q,
                self.sim_dt,
                element_sources=self.cutter.pending_geometry.source if self.streaming_topology else None,
            )
            if surface_count > self.contact.capacity:
                if self.reserve is not None:
                    raise ValueError(
                        f"boundary surface reserve exhausted: the cut needs {surface_count} triangles, "
                        f"{self.contact.capacity} are reserved"
                    )
                self._grow_surface(surface_count)
            else:
                self.contact.update_topology(self.cutter.surface, surface_count, self.state_0.particle_q)
            if self.device.is_cuda:
                wp.synchronize_device(self.device)
            self.cutter.commit()
            self.fragment_count, self._surface_count = diagnostics["fragment_count"], surface_count
            self.cut_revision += 1
        except (MemoryError, ValueError, RuntimeError) as error:
            self.cutter.discard()
            for grips in self.grip_sets[:staged_grips]:
                grips.restore()
            self.contact, self.surface, self._render_indices, self.graph = previous_surface_buffers
            if published:
                self._model_factory.copy_into(self.model, self._backup_model)
                self.state_0.assign(self._backup_state)
                self.state_1.assign(self._backup_state)
                self._update_mechanical_components(self.cutter.material_components)
                self.solver.refresh_component_scaling()
                self.solver.refresh_topology(self.state_0.particle_q, self.sim_dt)
                wp.copy(self.solver.velocity_increment, self._backup_velocity_increment)
                # The staged positions: the contact may hold the surface of the refused edit, whose
                # cloned nodes have no place in the restored state. The nodes of the restored surface
                # are where they were in both.
                self.contact.update_topology(self.cutter.surface, previous_surface_count, self._staged_state.particle_q)
            self.fragment_count, self._surface_count = previous_count, previous_surface_count
            self.cut_error = self._rejection_message(
                error, self.cutter.geometry.capacity, fixed=self.reserve is not None
            )
            return False
        self.last_cut_ms = 1000.0 * (perf_counter() - start)
        self.last_rebuild_ms = self.last_cut_ms - self.last_geometry_ms
        self.cut_error = ""
        return True

    def check_startups(self, startup, census_failure):
        """The check of a self-test: nothing captured or loaded after a start-up that storage growth did not force.

        Raises AssertionError on a CUDA graph capture or a Krylov solve outside a graph after a
        start-up mark (the construction and every Reset) that did not fall into a growth of
        storage, on a kernel module load after a mark and on a start-up graph that was never
        launched. The captures inside a growth are counted apart and listed with their growth:
        without ``reserve`` an edit may grow storage, and a growth binds new buffers, so the
        graphs on them are captured in that frame. With ``reserve`` nothing grows and every
        capture after a mark fails.

        Args:
            startup: The start-ups of the scene: its ``check()`` and ``census()`` give the numbers
                after the marks (the ``Startup`` of the examples).
            census_failure: What a census shows that must not be there, as text, or None.

        Returns:
            The numbers of the check, for the line that a self-test prints.
        """
        growths = [
            {
                "what": growth["what"],
                "from": growth["from"],
                "to": growth["to"],
                "frame": growth["frame"],
                "ms": round(1000.0 * growth["seconds"], 1),
                "captures": growth["captures"] + growth.get("solver_captures", 0),
            }
            for growth in self.storage_growths
        ]
        forced = sum(growth["captures"] for growth in self.storage_growths)
        forced_solver = sum(growth.get("solver_captures", 0) for growth in self.storage_growths)
        if self.reserve is not None and growths:
            raise AssertionError(f"Storage grew although the cut reserve is fixed: {growths}")
        if not growths:
            numbers = startup.check()
        else:
            census = startup.census()
            if census["capture_census_status"] != "checked":
                numbers = startup.check()
            else:
                beyond = dict(census)
                beyond["captures_after_startup"] -= forced
                beyond["solver_captures_after_startup"] -= forced_solver
                failure = census_failure(beyond)
                if failure is not None:
                    raise AssertionError(
                        "Graph captures or kernel module loads after start-up that no storage growth forced, or a "
                        f"start-up graph that was never launched: {failure}; storage growths {growths}"
                    )
                numbers = {name: value for name, value in census.items() if not isinstance(value, dict)}
        return {
            **numbers,
            "storage_growths": growths,
            "captures_forced_by_storage_growth": forced + forced_solver,
            "cut_reserve": None if self.reserve is None else self.reserve["scale"],
            # Scratch vertices, scratch faces and work entries of the subtractor as it stands.
            "subtractor_storage": list(self.cutter.subtractor_storage),
        }
