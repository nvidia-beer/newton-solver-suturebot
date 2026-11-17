# SPDX-License-Identifier: Apache-2.0
"""The dVRK tools and the FEM tissue block of the grasp scene, without needle, thread or coupler.

The tools are those of ``GraspScene`` on the same rigid solver (MuJoCo Warp, or native MuJoCo
on the CPU), the tissue is the region of ``fem_phantom_regions`` with the material and solver
settings of the grasp scene. Tools and tissue meet only through one tool/tissue contact
object (``_tool_tissue_contact.py``); without one the tools pass through the tissue.

One physics interval, in this order:

1. the reaction of the contact on the tools, stored at the last accepted interval, is added
   to the body forces (``contact.add_reaction``);
2. the rigid interval of the tools: four MuJoCo steps with those loads and the actuator
   targets held;
3. the contact adds its forces on the tissue, with the motion of the tools over the interval:
   their poses at its start and at its end (``contact.begin_step``);
4. the tissue step, after which a contact may correct the tissue's end state
   (``contact.correct_step``);
5. one check of the interval, read once by the host, which decides whether it is accepted.

An accepted interval ends with ``contact.end_step`` at the accepted state. A rejected one
(the tetrahedron guard of the coupled scenes) is repeated from the unchanged start with half
the interval; nothing of it is kept.

On a CUDA device with the MuJoCo Warp backend, steps 1 to 5 are one CUDA graph for each of
the two state buffers, the end-of-step call is another, and the tangent refresh of a frame a
third kind. :meth:`ToolTissueScene.capture_graphs` captures and launches all of them before
the first frame and leaves every array with the bits it had.
"""

import gc
import math

import numpy as np
import warp as wp

from ..._src.geometry.mesh_refit import GraphMeshRefit
from ..._src.solvers.coupled.interface.swept_tool_tissue_contact import capsule_pieces
from ..._src.solvers.coupled.interface.tool_tissue_contact import _find, capsule_samples
from ..._src.solvers.coupled.tet_step import _minimum_tet_motion
from ..._src.solvers.mujoco.experimental_rigid_backend import is_gpu_backend
from ..._src.solvers.mujoco.subcycled_mujoco_warp import WarpIntegrationCheckpoint
from ..._src.solvers.thread.tissue import _move_surface
from ..._src.utils import capture
from ..needle.example_fem_needle_insertion import RENDER_MODULE_PREFIXES
from ..phantom.example_fem_phantom_regions import Example as PhantomExample
from ._dvrk_layout import load_source_layout
from ._grasp_scene import (
    GraspScene,
    _ramp_arm_targets,
    add_dvrk_tools,
    tool_pose_ik,
    tool_rigid_solver,
)
from ._tool_tissue_contact import make_contact
from .parallel import SutureSceneConfig

# Opening of the parked second tool [rad], as the grasp scene parks it (GraspScene._initialize_pose).
PARKED_JAW_OPENING_RAD = 0.7


class TissueStepRejected(RuntimeError):
    """The tissue step of an interval did not pass the tetrahedron guard; nothing of it is kept."""


class SceneFailed(RuntimeError):
    """The scene cannot go on from its accepted state."""


@wp.kernel(enable_backward=False)
def _clear_loads(body_f: wp.array[wp.spatial_vector], particle_f: wp.array[wp.vec3], check: wp.array[wp.float64]):
    # External loads of the interval and the result of its check, without a memset in the graph.
    index = wp.tid()
    if index < body_f.shape[0]:
        body_f[index] = wp.spatial_vector()
    if index < particle_f.shape[0]:
        particle_f[index] = wp.vec3(0.0)
    if index < check.shape[0]:
        if index == 0:
            check[index] = wp.float64(wp.inf)
        else:
            check[index] = wp.float64(0.0)


@wp.func
def _gap_at_rest(surface: wp.uint64, centre: wp.vec3, radius: float, reach: float):
    """Gap of a sphere to the tissue surface at rest [m]: negative inside it, infinite beyond ``reach``."""
    query = wp.mesh_query_point_sign_normal(surface, centre, reach)
    if not query.result:
        return float(wp.inf)
    nearest = wp.mesh_eval_position(surface, query.face, query.u, query.v)
    return query.sign * wp.length(centre - nearest) - radius


@wp.kernel(enable_backward=False)
def _rigid_check(
    start: wp.array[wp.transform],
    end: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    tip_body: int,
    tip_local: wp.vec3,
    tip_radius: float,
    rest_surface: wp.uint64,
    reach: float,
    check: wp.array[wp.float64],
    track: wp.array[float],
):
    """Nonfinite rigid end state of an interval (check[1]) and the track of the jaw tip (``ToolTissueScene._tip_travel``).

    track[0] is the largest travel of the tip in one interval. An interval in which the gap of the
    tip sphere to the surface at rest changes sign, from clear of it to inside it, leaves its
    travel in track[2] and the gap at its start and at its end in track[3] and track[4].
    """
    body = wp.tid()
    position = wp.transform_get_translation(end[body])
    rotation = wp.transform_get_rotation(end[body])
    rate = velocity[body]
    invalid = False
    for axis in range(3):
        if not wp.isfinite(position[axis]):
            invalid = True
    for axis in range(4):
        if not wp.isfinite(rotation[axis]):
            invalid = True
    for axis in range(6):
        if not wp.isfinite(rate[axis]):
            invalid = True
    if invalid:
        wp.atomic_max(check, 1, wp.float64(1.0))
    if body == tip_body:
        before = wp.transform_point(start[body], tip_local)
        after = wp.transform_point(end[body], tip_local)
        moved = wp.length(after - before)
        wp.atomic_max(track, 0, moved)
        gap_before = _gap_at_rest(rest_surface, before, tip_radius, reach)
        gap_after = _gap_at_rest(rest_surface, after, tip_radius, reach)
        if gap_before > 0.0 and gap_after <= 0.0:
            track[2] = moved
            track[3] = gap_before
            track[4] = gap_after


@wp.kernel(enable_backward=False)
def _tip_gap(
    body_q: wp.array[wp.transform],
    tip_body: int,
    tip_local: wp.vec3,
    tip_radius: float,
    rest_surface: wp.uint64,
    reach: float,
    track: wp.array[float],
):
    # Gap of the tip sphere to the surface at rest at the observed state (track[1]).
    track[1] = _gap_at_rest(rest_surface, wp.transform_point(body_q[tip_body], tip_local), tip_radius, reach)


@wp.kernel(enable_backward=False)
def _overflow_check(overflow: wp.array[int], check: wp.array[wp.float64]):
    # Capacity or solver-budget overflow of MuJoCo Warp in any step of the interval.
    if overflow[wp.tid()] != 0:
        wp.atomic_max(check, 2, wp.float64(1.0))


@wp.kernel(enable_backward=False)
def _begin_observation(depth: wp.array[float], displacement: wp.array[float], capsule_depth: wp.array2d[float]):
    index = wp.tid()
    if index < depth.shape[0]:
        depth[index] = -wp.inf
        for kind in range(capsule_depth.shape[1]):
            capsule_depth[index, kind] = 0.0
    if index < displacement.shape[0]:
        displacement[index] = 0.0


@wp.kernel(enable_backward=False)
def _capsule_depths(
    mesh: wp.uint64,
    body_q: wp.array[wp.transform],
    point_body: wp.array[int],
    point_local: wp.array[wp.vec3],
    point_radius: wp.array[float],
    triangles: wp.array[int],
    vertex_held: wp.array[int],
    reach: float,
    first: int,
    depth: wp.array2d[float],
):
    """How far the drawn capsules are inside the tissue [m], per body: independent of any contact law.

    Two kinds of points, each asked for the nearest point of the closed tissue surface and
    for its side of it (ray parity):

    - A point of the capsule surface (``point_radius`` zero) that is inside the tissue is as
      deep as its distance d from the surface. Between such points the capsule is not
      sampled, so this reads low.
    - A station of the axis (``point_radius`` the radius R of the capsule). Under the
      surface, its distance is the depth of the centre line. Outside it, the capsule about
      it reaches R - d into the tissue, and no point of the capsule is further inside than
      the largest R - d over the axis: a point p of the capsule that is delta inside lies
      within R of an axis point a, the way from a to p crosses the surface at least d(a)
      from a and at least delta from p, so delta <= R - d(a). This is exact around the axis
      and on the caps.

    ``depth[body, first]`` is the largest of both where the nearest triangle can yield,
    ``first + 1`` where it has a vertex that is fixed or outside the simulated block,
    ``first + 2`` the deepest station of the centre line.
    """
    index = wp.tid()
    body = point_body[index]
    point = wp.transform_point(body_q[body], point_local[index])
    query = wp.mesh_query_point(mesh, point, reach)
    if not query.result:
        return
    inside = wp.length(wp.mesh_eval_position(mesh, query.face, query.u, query.v) - point)
    if point_radius[index] > 0.0:
        if query.sign < 0.0:
            wp.atomic_max(depth, body, first + 2, inside)
            return
        inside = point_radius[index] - inside
        if inside <= 0.0:
            return
    elif query.sign >= 0.0:
        return
    held = int(0)
    for corner in range(3):
        held += vertex_held[triangles[3 * query.face + corner]]
    if held > 0:
        wp.atomic_max(depth, body, first + 1, inside)
    else:
        wp.atomic_max(depth, body, first, inside)


@wp.kernel(enable_backward=False)
def _part_depths(gap: wp.array[float], face: wp.array[int], sample_body: wp.array[int], depth: wp.array[float]):
    """Deepest point of the samples of every body below the surface [m]: positive inside, negative clear of it."""
    sample = wp.tid()
    if face[sample] >= 0:
        wp.atomic_max(depth, sample_body[sample], -gap[sample])


@wp.kernel(enable_backward=False)
def _surface_displacement(
    q: wp.array[wp.vec3],
    rest: wp.array[wp.vec3],
    nodes: wp.array[int],
    normals: wp.array[wp.vec3],
    displacement: wp.array[float],
):
    """Largest displacement of a surface node [m] (entry 0) and its largest part into the tissue (entry 1)."""
    node = nodes[wp.tid()]
    moved = q[node] - rest[node]
    wp.atomic_max(displacement, 0, wp.length(moved))
    wp.atomic_max(displacement, 1, -wp.dot(moved, normals[wp.tid()]))


def _conjugate(quaternion):
    """The inverse of a unit quaternion (xyzw)."""
    return np.r_[-np.asarray(quaternion[:3], dtype=float), float(quaternion[3])]


def _rotate(quaternion, points):
    """Points turned by a unit quaternion (xyzw), on the host."""
    vector, scalar = np.asarray(quaternion[:3], dtype=float), float(quaternion[3])
    return points + 2.0 * np.cross(vector, np.cross(vector, points) + scalar * points)


class TissueSurface:
    """The exterior surface of the whole pad as a ``wp.Mesh``; its vertices in the tissue block follow the particles.

    The surface reaches beyond the simulated block, as the surface of the thread and of the
    first tool/tissue contact in the grasp scene does: a vertex that is no particle stays at
    rest. :meth:`update` moves the particles' vertices and refits the mesh, with launches only.

    Attributes:
        mesh: The surface, outward-facing triangles.
        triangles: Its triangles as flat vertex indices.
        point_particle: Tissue particle of each vertex, -1 for a vertex that is not one.
    """

    def __init__(self, vertices, triangles, particle_points, device):
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        particle_points = np.asarray(particle_points, dtype=np.int32)
        point_particle = np.full(len(vertices), -1, dtype=np.int32)
        point_particle[particle_points] = np.arange(len(particle_points))
        self.device = device
        self.triangles = wp.array(np.asarray(triangles, dtype=np.int32).reshape(-1), dtype=int, device=device)
        self.point_particle = wp.array(point_particle, dtype=int, device=device)
        self._particle_point = wp.array(particle_points, dtype=int, device=device)
        self._points = wp.array(vertices, dtype=wp.vec3, device=device)
        self.mesh = wp.Mesh(self._points, self.triangles)
        # Warp's own refit allocates; the graph-safe refit exists for CUDA only and skips what cannot move.
        self._refit = GraphMeshRefit(self.mesh, moving_vertex_mask=point_particle >= 0) if device.is_cuda else None

    def update(self, particle_q):
        """Move the surface to the tissue at ``particle_q`` and refit it."""
        wp.launch(
            _move_surface,
            dim=len(self._particle_point),
            inputs=[particle_q, self._particle_point],
            outputs=[self._points],
            device=self.device,
        )
        if self._refit is not None:
            self._refit.refit()
        else:
            self.mesh.refit()


class ToolTissueScene(PhantomExample):
    """Two dVRK tools and the tissue block; the first tool follows arm targets, the second is parked.

    Args:
        viewer: Viewer of the phantom example this scene is built on; a null viewer.
        args: Parsed arguments of :meth:`create_parser`.
        rigid_backend: "mujoco-warp" (CUDA) or "native" (CPU MuJoCo).
        contact: Name of the tool/tissue contact (``_tool_tissue_contact.CONTACTS``).

    The owner of the scene places the first tool (:meth:`place`), sets the arm targets of a
    frame (:meth:`set_frame_targets`) and advances a frame (:meth:`step`). :meth:`observe` is the
    only call that reads the state back; the step reads one small array per interval.
    """

    def __init__(self, viewer, args, *, rigid_backend="mujoco-warp", contact="off"):
        if rigid_backend not in ("native", "mujoco-warp"):
            raise ValueError("rigid_backend must be native or mujoco-warp")
        self.rigid_backend = rigid_backend
        self.contact_name = contact
        self.contact = None
        self.rigid_solver = None
        self._startup_mark = None
        self._startup_ready = False
        self._startup_status = "start-up not marked"
        # Everything from here to the mark is start-up (finish_startup).
        self._startup_begun = capture.begin_startup()
        super().__init__(viewer, args)

    @staticmethod
    def create_parser():
        """The phantom example's options with the tissue of the grasp scene.

        Material, region and solver settings are read from the parser of the grasp scene and
        the physics intervals per frame from its configuration, so the block is the same by
        construction.
        """
        parser = PhantomExample.create_parser()
        grasp = GraspScene.create_parser()
        names = (
            "regions",
            "solver_iterations",
            "young_modulus",
            "poisson_ratio",
            "density",
            "material_damping",
            "linear_damping",
        )
        parser.set_defaults(
            **{name: grasp.get_default(name) for name in names}, substeps=SutureSceneConfig(device="cpu").substeps
        )
        parser.add_argument(
            "--fem-tolerance",
            type=float,
            default=grasp.get_default("fem_tolerance"),
            help="Relative FEM linear residual tolerance",
        )
        parser.add_argument(
            "--fem-chebyshev-degree",
            type=int,
            default=grasp.get_default("fem_chebyshev_degree"),
            help="Chebyshev acceleration degree of the FEM block-Jacobi preconditioner; 0 disables",
        )
        return parser

    # ------------------------------------------------------------------ construction

    def configure_builder(self, builder):
        self.source_layout = load_source_layout()
        self.graspers = add_dvrk_tools(builder)

    def _capture_physics(self):
        # The graphs of this scene are captured by capture_graphs(), with the tools in place.
        self.graph = None

    def select_regions(self, region_ids):
        super().select_regions(region_ids)
        if self.solver is None:
            raise ValueError("The tool/tissue scene needs one active tissue region")
        device = self.device
        self.frame_dt = SutureSceneConfig(device="cpu").frame_dt
        self.sim_dt = self.frame_dt / self.args.substeps
        # As the needle insertion example configures the tissue solver of the grasp scene.
        self.solver.linear_solver_tolerance = self.args.fem_tolerance
        self.solver.linear_solver_chebyshev_degree = self.args.fem_chebyshev_degree
        self.control = self.model.control()
        self._external_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self._frame_particle_q = wp.empty_like(self.model.particle_q)
        self._saved_increment = wp.empty_like(self.solver.velocity_increment)
        # [0] smallest swept volume bound of a tetrahedron, [1] nonfinite rigid state, [2] rigid overflow.
        self._check = wp.zeros(3, dtype=wp.float64, device=device)
        # Track of the jaw tip since the last observation: [0] largest travel in one interval [m], [1] gap of the
        # tip sphere to the surface at rest at the observed state [m], [2] travel in the interval in which that gap
        # changed sign from clear to inside (0: none), [3] and [4] the gap at the start and at the end of it.
        self._tip_travel = wp.zeros(5, dtype=float, device=device)
        self.elapsed = self.sim_time = 0.0
        self.frames = self.accepted_intervals = self.rejected_intervals = self.failed_intervals = 0
        self.minimum_step_dt = self.sim_dt
        # Why the scene stopped, and whether the tissue step ("tissue") or the rigid solver ("rigid") did not go on.
        self.failure_reason = self.failure_kind = None
        self._graphs, self._end_graphs, self._tangent_graph = {}, {}, None
        self._ramp = None
        self._build_tools()
        self._build_surface()
        self.contact = make_contact(self.contact_name, self)

    def _build_tools(self):
        """Place the fixed trocars, build the rigid solver and the inverse kinematics, park the second tool."""
        for index, tool in enumerate(self.graspers):
            tool.set_trocar_pose(self.model, self.source_layout["trocar_poses"][index], reset=True)
            tool.initialize(self.model, self.state_0, opening_rad=PARKED_JAW_OPENING_RAD if index else 0.0)
        options = {}
        if self.rigid_backend == "mujoco-warp":
            # No trial loop repeats an interval here: every step starts from MuJoCo's own warm start.
            options["trial_warm_start"] = False
        self.rigid_solver = tool_rigid_solver(self.model, self.rigid_backend, **options)
        self.rigid_solver.reset(self.state_0, flags=0)
        self.pose_ik = tool_pose_ik(self.model, self.rigid_solver, self.graspers)
        self.gpu_rigid = is_gpu_backend(self.rigid_solver)
        self._checkpoint = WarpIntegrationCheckpoint(self.rigid_solver) if self.gpu_rigid else None
        self._native_state = None
        if self.gpu_rigid:
            # One step time for the rigid interval and the tissue step: one host write per interval.
            self.rigid_solver.adopt_step_time(self.solver.step_time)
        self.parked_target = self.source_layout["parked_pose"].copy()
        parked = self.pose_ik[1].solve(self.parked_target, self.rigid_solver.mj_data.qpos)
        self.graspers[1].initialize(self.model, self.state_0, arm_positions=parked, opening_rad=PARKED_JAW_OPENING_RAD)
        self._parked_arm = np.asarray(parked, dtype=float)
        # The one capsule set of the tools, as their models publish it: the contact, the measurements and the
        # view read these rows.
        bodies, endpoints, radii = [], [], []
        for tool in self.graspers:
            bodies.extend(tool.capsule_bodies)
            endpoints.extend(tool.capsule_endpoints)
            radii.extend(tool.capsule_radii)
        self.capsule_bodies, self.capsule_endpoints = np.asarray(bodies, dtype=np.int32), np.asarray(endpoints)
        self.capsule_radii = np.asarray(radii, dtype=float)
        # The radii with which the capsules touch the tissue are the radii of the set itself.
        self.capsule_contact_radii = self.capsule_radii
        # The spheres of the capsules, as the first tool/tissue contact samples them.
        self.sample_bodies, self.sample_points, self.sample_radii = capsule_samples(
            self.capsule_bodies, self.capsule_endpoints, self.capsule_radii
        )
        # Name of the tool part on every body that carries capsules (a part has one or more capsules).
        self.part_names = {
            int(body): f"PSM{index + 1}/{name}"
            for index, tool in enumerate(self.graspers)
            for body, name in zip(tool.capsule_bodies, tool.capsule_parts, strict=True)
        }
        tool = self.graspers[0]
        # Closed jaws share the frame of the palm: their spheres are fixed in it.
        jaws = np.isin(self.sample_bodies, tool.jaw_bodies)
        self.jaw_points, self.jaw_radii = self.sample_points[jaws], self.sample_radii[jaws]
        # The tip: the sphere of the first jaw that lies farthest along the jaw.
        first = np.flatnonzero(self.sample_bodies == tool.jaw_bodies[0])
        self.tip_body = int(tool.jaw_bodies[0])
        tip = first[np.argmax(self.sample_points[first, 1])]
        self.tip_local = self.sample_points[tip].copy()
        self.tip_radius = float(self.sample_radii[tip])
        # The radius of the jaws where they lead into the tissue: that of the capsule that carries the tip.
        self.jaw_contact_radius = self.tip_radius
        gains = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
        dofs = list(tool.arm_dof_indices)
        # Lag of a position actuator behind a ramp of its target: speed times kd / kp.
        self.actuator_time_constants = gains[1][dofs] / gains[0][dofs]
        # The jaws that open: the jaw of every jaw sphere, and the hinge of each jaw in the frame of the palm
        # (frame of the hinge on the palm, its axis, frame of the hinge on the jaw), as the tool model has them.
        self.jaw_of_sphere = (self.sample_bodies[jaws] == tool.jaw_bodies[1]).astype(np.int32)
        joints, hinge_dofs = list(tool.jaw_joints), list(tool.jaw_dof_indices)
        self._jaw_hinges = (
            self.model.joint_X_p.numpy()[joints].astype(float),
            self.model.joint_axis.numpy()[hinge_dofs].astype(float),
            self.model.joint_X_c.numpy()[joints].astype(float),
        )
        # Largest half-opening of a jaw [rad]: the limit of its hinge.
        self.maximum_half_opening = 0.5 * float(tool.config.maximum_opening_rad)

    def _build_surface(self):
        device = self.device
        reference, selection = self.reference, self.selection
        ids = np.searchsorted(reference.global_vertex_indices, selection.global_vertex_indices)
        if np.any(ids >= len(reference.vertices)) or not np.array_equal(
            reference.global_vertex_indices[ids], selection.global_vertex_indices
        ):
            raise ValueError("Every tissue particle must be a vertex of the complete pad")
        self.surface = TissueSurface(reference.vertices, reference.exterior_surface_triangles, ids, device)
        self.surface_reach = float(np.ptp(np.asarray(reference.vertices)[:, 2]))
        # The same surface at rest, never moved: where the tissue surface was before a tool touched it. The gap
        # of the jaw tip to it tells whether, and in which interval, the tool crossed that surface.
        self._rest_points = wp.array(np.asarray(reference.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
        self.rest_surface = wp.Mesh(self._rest_points, self.surface.triangles)
        rest = np.asarray(selection.vertices, dtype=float)
        triangles = np.asarray(selection.exterior_surface_triangles)
        normals = np.zeros_like(rest)
        faces = np.cross(rest[triangles[:, 1]] - rest[triangles[:, 0]], rest[triangles[:, 2]] - rest[triangles[:, 0]])
        for corner in range(3):
            np.add.at(normals, triangles[:, corner], faces)
        nodes = np.unique(triangles)
        lengths = np.linalg.norm(normals[nodes], axis=1)
        self.surface_nodes = nodes
        self.surface_normals = normals[nodes] / lengths[:, None]
        tetrahedra = np.asarray(selection.indices)
        edges = rest[tetrahedra[:, 1:]] - rest[tetrahedra[:, :1]]
        # Mean tetrahedron edge [m], the mesh spacing that the thread and the first contact use.
        self.mesh_spacing = float(np.linalg.norm(edges, axis=-1).mean())
        self._rest = wp.array(rest, dtype=wp.vec3, device=device)
        self._surface_nodes = wp.array(nodes, dtype=int, device=device)
        self._surface_normals = wp.array(self.surface_normals, dtype=wp.vec3, device=device)
        count = len(self.sample_bodies)
        self._sample_body = wp.array(self.sample_bodies, dtype=int, device=device)
        self._sample_local = wp.array(self.sample_points, dtype=wp.vec3, device=device)
        self._sample_radius = wp.array(self.sample_radii, dtype=float, device=device)
        self._sample_scratch = (
            wp.zeros(count, dtype=float, device=device),  # force: no sample of an observation carries one
            wp.zeros(count, dtype=float, device=device),  # gap
            wp.full(count, -1, dtype=int, device=device),  # face
            wp.zeros(count, dtype=wp.vec3, device=device),
            wp.zeros(count, dtype=wp.vec3, device=device),
            wp.zeros(count, dtype=wp.vec3, device=device),
            wp.zeros(1, dtype=int, device=device),
        )
        self._part_depth = wp.zeros(self.model.body_count, dtype=float, device=device)
        self._displacement = wp.zeros(2, dtype=float, device=device)
        self._build_capsule_points()

    def _build_capsule_points(self):
        """Points of the capsules for the measurement of how far each of them is inside the tissue.

        Per capsule, with its radius R and its axis from the first to the last contact
        sphere (``_capsule_depths`` says what is measured on each kind):

        - stations of the axis at most R / 32 apart, with both ends. While the axis is outside
          the tissue they give the depth of the capsule exactly around the axis and on the
          caps. Along the axis the surface point nearest to it lies across it from an axis
          point at most R / 64 from a station (or at an end, which is a station), so its
          distance d is read at most (R / 64)^2 / (2 d) too large: less than 0.25 micrometres
          too shallow for a capsule of a millimetre radius that is less than half a radius inside.
        - points of the capsule surface: rings of 48 every R / 8 along the axis and eight
          rings on each cap. They alone say how deep a capsule is whose axis is under the
          surface; they read low by what lies between two points (the check of this example
          measured up to 10 micrometres against a sampling twice as fine).
        """
        device = self.device
        _, _, starts, ends, radii, _, firsts, counts = capsule_pieces(
            self.capsule_bodies, self.capsule_endpoints, self.capsule_radii, self.capsule_contact_radii
        )
        angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        bodies, points, point_radii = [], [], []
        for body, first, count in zip(self.capsule_bodies, firsts, counts, strict=True):
            start, end, radius = starts[first], ends[first + count - 1], float(radii[first])
            length = float(np.linalg.norm(end - start))
            axis = (end - start) / length
            side = np.cross(axis, (0.0, 0.0, 1.0))
            if np.linalg.norm(side) < 0.5:
                side = np.cross(axis, (0.0, 1.0, 0.0))
            side /= np.linalg.norm(side)
            ring = np.cos(angles)[:, None] * side + np.sin(angles)[:, None] * np.cross(axis, side)
            rings = start + np.linspace(0.0, 1.0, int(math.ceil(8.0 * length / radius)) + 1)[:, None] * (end - start)
            surface = [(rings[:, None, :] + radius * ring[None, :, :]).reshape(-1, 3)]
            for tip, sign in ((start, -1.0), (end, 1.0)):
                for polar in np.linspace(0.0, 0.5 * np.pi, 9)[1:]:
                    surface.append(tip + radius * (np.cos(polar) * ring + np.sin(polar) * sign * axis))
            surface = np.concatenate(surface)
            fractions = np.linspace(0.0, 1.0, int(math.ceil(32.0 * length / radius)) + 1)
            stations = start + fractions[:, None] * (end - start)
            bodies.extend([int(body)] * (len(surface) + len(stations)))
            points.extend((surface, stations))
            point_radii.extend([0.0] * len(surface) + [radius] * len(stations))
        particle = self.surface.point_particle.numpy()
        fixed = np.zeros(self.model.particle_count, dtype=bool)
        fixed[np.asarray(self.fixed_vertex_indices)] = True
        self._capsule_point_body = wp.array(np.asarray(bodies, dtype=np.int32), dtype=int, device=device)
        self._capsule_point_local = wp.array(np.concatenate(points), dtype=wp.vec3, device=device)
        self._capsule_point_radius = wp.array(np.asarray(point_radii), dtype=float, device=device)
        # Vertices of the pad surface that cannot move: fixed particles and vertices outside the simulated block.
        self._vertex_held = wp.array(
            ((particle < 0) | fixed[np.maximum(particle, 0)]).astype(np.int32), dtype=int, device=device
        )
        # Per body: yielding, held and centre-line depth at the accepted state, the same three as drawn, and
        # yielding and held depth of the drawn parts themselves (their CAD vertices) as drawn.
        self._capsule_depth = wp.zeros((self.model.body_count, 9), dtype=float, device=device)
        self.capsule_point_count = len(bodies)
        self._build_drawn_points()
        # Hook of a measurement: called after every accepted interval (and its end-of-step call) with the
        # interval [s]. None in a normal run: the step then reads nothing but its own check.
        self.interval_observer = None
        # Actuator gains and force limits of the joints, for the demand of the position actuators.
        self._actuator_ke = self.model.joint_target_ke.numpy().astype(float)
        self._actuator_kd = self.model.joint_target_kd.numpy().astype(float)
        self._actuator_limit = self.model.joint_effort_limit.numpy().astype(float)

    def _build_drawn_points(self):
        """Vertices of the drawn tool parts (the CAD that the viewer draws) on their bodies.

        They are measured like the surface points of the capsules (``_capsule_depths``): a vertex
        inside the closed tissue surface is as deep as its distance from it. This is how far the
        tool that is seen is inside the tissue that is seen, whatever the collision shapes are.
        """
        from ._grasp_visual_assets import load_dvrk_tool, physical_tool_parts

        device = self.device
        bodies, points, self.drawn_part_names = [], [], {}
        for index, tool in enumerate(self.graspers):
            instrument = f"PSM{index + 1}"
            asset = load_dvrk_tool(instrument=instrument, jaw_opening=0, shaft_length_m=tool.config.shaft_length_m)
            parts = physical_tool_parts(
                asset,
                {"instrument": instrument, "part_body_indices": tool.part_body_indices},
                body_count=self.model.body_count,
            )
            for body, mesh in parts:
                vertices = np.asarray(mesh.vertices, dtype=float)
                bodies.extend([int(body)] * len(vertices))
                points.append(vertices)
                self.drawn_part_names[int(body)] = f"{instrument}/{mesh.name}"
        self._drawn_point_body = wp.array(np.asarray(bodies, dtype=np.int32), dtype=int, device=device)
        self._drawn_point_local = wp.array(np.concatenate(points), dtype=wp.vec3, device=device)
        self._drawn_point_radius = wp.zeros(len(bodies), dtype=float, device=device)
        self.drawn_point_count = len(bodies)

    # ------------------------------------------------------------------ planning helpers (start-up only)

    def jaw_spheres(self, half_opening):
        """Contact spheres of the two jaws of the first tool in the frame of its palm, with the jaws opened.

        Each jaw is turned about its hinge by ``half_opening`` [rad] away from the other, as
        ``GrasperAssembly.set_aperture`` commands it (the first jaw by the angle, the second by
        its negative), with the hinge frames and axes of the tool model. For planning, on the host.

        Returns:
            Centres [m], shape (spheres, 3), and radii [m] of the samples of the jaws' capsules, in
            the order of :attr:`jaw_points`; :attr:`jaw_of_sphere` names the jaw of each.
        """
        parents, axes, children = self._jaw_hinges
        points = np.empty_like(self.jaw_points)
        for jaw, sign in enumerate((1.0, -1.0)):
            mine = self.jaw_of_sphere == jaw
            axis = axes[jaw] / np.linalg.norm(axes[jaw])
            # In the frame of the hinge on the jaw, turned about the hinge axis (Rodrigues), in the frame of the palm.
            local = _rotate(_conjugate(children[jaw, 3:]), self.jaw_points[mine] - children[jaw, :3])
            angle = sign * float(half_opening)
            turned = (
                local * math.cos(angle)
                + np.cross(axis, local) * math.sin(angle)
                + np.outer(local @ axis, axis) * (1.0 - math.cos(angle))
            )
            points[mine] = _rotate(parents[jaw, 3:], turned) + parents[jaw, :3]
        return points, self.jaw_radii

    def jaw_depth_at_rest(self, positions, rotation, half_opening=0.0):
        """Depth of the jaws of the first tool below the surface that :attr:`surface` holds [m].

        For palm positions [m], shape (count, 3), with one palm rotation (xyzw): the deepest point
        of the jaw spheres, positive inside the tissue and negative clear of it; minus infinity
        where no surface is within reach. The jaws are closed, or opened by ``half_opening`` [rad]
        each (:meth:`jaw_spheres`). One launch for all positions and one read: for planning,
        before the run, with the surface at rest.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        jaw_points = self.jaw_spheres(half_opening)[0] if half_opening else self.jaw_points
        count, samples = len(positions), len(jaw_points)
        device = self.device
        poses = np.concatenate((positions, np.tile(np.asarray(rotation, dtype=float), (count, 1))), axis=1)
        gap = wp.zeros(count * samples, dtype=float, device=device)
        face = wp.full(count * samples, -1, dtype=int, device=device)
        scratch = [wp.zeros(count * samples, dtype=wp.vec3, device=device) for _ in range(3)]
        wp.launch(
            _find,
            count * samples,
            inputs=[
                self.surface.mesh.id,
                wp.array(poses.astype(np.float32), dtype=wp.transform, device=device),
                wp.array(np.repeat(np.arange(count, dtype=np.int32), samples), dtype=int, device=device),
                wp.array(np.tile(jaw_points, (count, 1)), dtype=wp.vec3, device=device),
                wp.array(np.tile(self.jaw_radii, count), dtype=float, device=device),
                self.surface_reach,
                wp.zeros(count * samples, dtype=float, device=device),
            ],
            outputs=[gap, face, *scratch, wp.zeros(1, dtype=int, device=device)],
            device=device,
        )
        depth = np.where(face.numpy() >= 0, -gap.numpy(), -np.inf).reshape(count, samples)
        return depth.max(axis=1)

    def solve_arm_targets(self, poses, *, initial=None):
        """Arm coordinates of the first tool for a sequence of palm poses, by its bounded inverse kinematics.

        Solved in order, each from the solution before it; the poses of a path are solved once,
        before the run. ``initial`` starts the first of them from given arm coordinates instead
        of from the last solution. Raises ``UnreachablePoseError`` for a pose outside the joint limits.
        """
        ik = self.pose_ik[0]
        qpos = self.rigid_solver.mj_data.qpos
        poses = np.asarray(poses, dtype=float)
        if initial is None or not len(poses):
            return np.asarray([ik.solve(pose, qpos) for pose in poses])
        first = ik.solve(poses[0], qpos, initial=initial)
        return np.asarray([first, *(ik.solve(pose, qpos) for pose in poses[1:])])

    def actuator_compliance(self, arm_positions, half_opening):
        """How far the palm of the first tool gives under loads on its bodies, at a planned configuration.

        With the position actuators of the tool as its only springs (joint stiffness ``K`` =
        the actuator gains of the model), wrenches ``w_b`` on its bodies move the palm by
        ``J_palm K^-1 sum_b J_b^T w_b`` once the tool has come to rest, ``J_b`` the Jacobian
        of body b at its centre of mass in the native model. A host computation on native data
        of its own, for planning: it is made at start-up, at the arm coordinates and the jaw
        opening that the path commands, not at a state of the run.

        Args:
            arm_positions: Arm coordinates of the first tool.
            half_opening: Half-opening of each of its jaws [rad].

        Returns:
            A function of a mapping from a body of the first tool (its palm or a jaw) to the
            wrench on it (force [N], then torque about its centre of mass [N m], world frame):
            the displacement of the palm [m], shape (3,).
        """
        import mujoco

        tool, solver = self.graspers[0], self.rigid_solver
        native = solver.mj_model
        state = self.model.state()
        tool.initialize(
            self.model,
            state,
            arm_positions=np.asarray(arm_positions, dtype=float),
            opening_rad=2.0 * float(half_opening),
        )
        data = mujoco.MjData(native)
        solver._update_mjc_data(data, self.model, state)
        mujoco.mj_forward(native, data)
        of_body = solver.mjc_body_to_newton.numpy()[0]
        # Stiffness of the position actuator of every degree of freedom [N/m or N m/rad]; zero without one.
        stiffness = np.zeros(native.nv)
        for actuator in range(native.nu):
            stiffness[native.jnt_dofadr[native.actuator_trnid[actuator, 0]]] = native.actuator_gainprm[actuator, 0]
        jacobians = {}
        for body in (tool.palm_body, *tool.jaw_bodies):
            index = int(np.flatnonzero(of_body == int(body))[0])
            translation, rotation = np.zeros((3, native.nv)), np.zeros((3, native.nv))
            mujoco.mj_jacBodyCom(native, data, translation, rotation, index)
            jacobians[int(body)] = np.concatenate((translation, rotation))
        moved = np.any(np.concatenate([np.abs(jacobian) for jacobian in jacobians.values()]) > 0.0, axis=0)
        if np.any(stiffness[moved] <= 0.0):
            raise ValueError("A joint of the first tool has no position actuator: its compliance is not bounded")
        give = np.where(stiffness > 0.0, 1.0 / np.where(stiffness > 0.0, stiffness, 1.0), 0.0)
        index = int(np.flatnonzero(of_body == int(tool.palm_body))[0])
        palm = np.zeros((3, native.nv))
        mujoco.mj_jac(native, data, palm, None, data.xpos[index], index)

        def palm_displacement(wrenches):
            torques = np.zeros(native.nv)
            for body, wrench in wrenches.items():
                torques += jacobians[int(body)].T @ np.asarray(wrench, dtype=float)
            return palm @ (give * torques)

        return palm_displacement

    # ------------------------------------------------------------------ reset-time placement

    def place(self, arm_positions, opening=0.0):
        """Put the first tool at arm coordinates with zero velocity; the tissue is left as it is.

        Its jaws are closed, or each opened by the half-opening ``opening`` [rad], in the state
        and in the actuator targets. A reset-time operation, like the placement of the grasp
        scene: joint coordinates are written, the rigid solver is reset to them and the actuator
        targets hold them.
        """
        arm = np.asarray(arm_positions, dtype=float)
        tools = self.graspers
        opening = float(opening)
        tools[0].initialize(self.model, self.state_0, arm_positions=arm, opening_rad=2.0 * opening)
        tools[1].initialize(
            self.model, self.state_0, arm_positions=self._parked_arm, opening_rad=PARKED_JAW_OPENING_RAD
        )
        self.rigid_solver.reset(self.state_0, flags=0)
        for tool, ik, target in zip(tools, self.pose_ik, (arm, self._parked_arm), strict=True):
            ik.reset(self.rigid_solver.mj_data.qpos)
            tool.set_arm_targets(self.control, target)
        self.set_jaw_opening(opening)
        tools[1].set_aperture(self.control, PARKED_JAW_OPENING_RAD / tools[1].config.maximum_opening_rad)
        if self.gpu_rigid:
            self.rigid_solver.initialize_from_newton(self.state_0, self.control)
        self.state_1.assign(self.state_0)
        self.set_frame_targets(arm, arm)
        # A placement is no travel: the track of the tip starts again.
        self._tip_travel.zero_()
        # What the contact takes from the configuration of the tools (host work, outside the step).
        self.contact.placed()

    def set_jaw_opening(self, half_opening):
        """Command the jaws of the first tool to open by ``half_opening`` [rad] each (zero: closed).

        The tool model's own aperture call (``GrasperAssembly.set_aperture``): two fills of the
        actuator targets on the device, no read. The targets hold until the next call.
        """
        self.graspers[0].set_aperture(self.control, float(half_opening) / self.maximum_half_opening)

    def reset_tissue(self):
        """Put the tissue back at rest and clear the contact; the tools stay where they are."""
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(self._rest)
            state.particle_qd.zero_()
        self.solver.velocity_increment.zero_()
        self.contact.reset()
        self.failure_reason = self.failure_kind = None

    def set_frame_targets(self, previous, target):
        """Arm targets of the first tool at the start and at the end of the next frame.

        The targets of an interval lie on the straight line between them, at the end of that
        interval (the ramp of the grasp scene).
        """
        previous, target = (np.asarray(value, dtype=np.float32) for value in (previous, target))
        if self._ramp is None:
            device = self.control.joint_target_q.device
            self._ramp = (
                wp.array(previous, dtype=float, device=device),
                wp.array(target, dtype=float, device=device),
                wp.array(list(self.graspers[0].arm_q_indices), dtype=int, device=device),
            )
        else:
            self._ramp[0].assign(previous)
            self._ramp[1].assign(target)

    # ------------------------------------------------------------------ one interval

    def _issue_interval(self, state_in, state_out, dt):
        """The launches of one interval from ``state_in`` into ``state_out``; eager, or into an open capture."""
        device = self.device
        if self._checkpoint is not None:
            # A rejected interval restarts from this integration state of MuJoCo Warp.
            self._checkpoint.save()
        wp.copy(self._saved_increment, self.solver.velocity_increment)
        wp.launch(
            _clear_loads,
            max(self.model.body_count, self.model.particle_count),
            [state_in.body_f, self._external_forces, self._check],
            device=device,
        )
        # --- Tool/tissue contact: its reaction reaches the tools here, as body forces of the rigid interval.
        self.contact.add_reaction(state_in.body_f)
        if self.gpu_rigid:
            self.rigid_solver._step_uncaptured(state_in, state_out, self.control, None, dt)
        else:
            self.rigid_solver.step(state_in, state_out, self.control, None, dt)
        # --- Tool/tissue contact: before the tissue step, with the motion of the tools over the interval
        # (state_in.body_q to state_out.body_q) and the tissue at its start; adds nodal forces [N].
        self.contact.begin_step(state_in, state_out, self._external_forces, dt)
        self.solver.step(state_in, state_out, self.control, None, dt, external_forces=self._external_forces)
        # --- Tool/tissue contact: after the tissue step and before the check, for a contact that corrects
        # the end state of the tissue; what it writes is checked with the step and discarded with it.
        self.contact.correct_step(state_in, state_out, dt)
        wp.launch(
            _minimum_tet_motion,
            self.model.tet_count,
            [
                state_in.particle_q,
                state_out.particle_q,
                self.model.tet_indices,
                self.model.tet_poses,
                self.model.tet_materials,
                self._check,
            ],
            device=device,
        )
        wp.launch(
            _rigid_check,
            self.model.body_count,
            [
                state_in.body_q,
                state_out.body_q,
                state_out.body_qd,
                self.tip_body,
                wp.vec3(*self.tip_local),
                self.tip_radius,
                self.rest_surface.id,
                self.surface_reach,
                self._check,
                self._tip_travel,
            ],
            device=device,
        )
        if self.gpu_rigid:
            wp.launch(
                _overflow_check,
                self.rigid_solver.mjw_data.nworld,
                [self.rigid_solver.mjw_data.overflow, self._check],
                device=device,
            )

    def _issue_end_step(self, accepted, dt):
        # --- Tool/tissue contact: after the interval is accepted, at the accepted state.
        self.contact.end_step(accepted, dt)

    def _restore_attempt(self):
        """Return the integration state of the rigid solver and the tissue solver's warm start to the interval's start."""
        if self._checkpoint is not None:
            self._checkpoint.restore()
        elif self._native_state is not None:
            import mujoco

            solver = self.rigid_solver
            mujoco.mj_setState(solver.mj_model, solver.mj_data, self._native_state, mujoco.mjtState.mjSTATE_INTEGRATION)
        wp.copy(self.solver.velocity_increment, self._saved_increment)

    def _attempt(self, dt):
        """One interval from the accepted state into the other buffer; raises if it is not accepted."""
        state_in, state_out = self.state_0, self.state_1
        graph = self._graphs.get(id(state_in))
        if graph is not None:
            # Host preparation of a replay: the graph reads the interval from the step time.
            self.solver.step_time.set(dt)
            self.rigid_solver.mj_model.opt.timestep = dt / self.rigid_solver.rigid_substeps
            wp.capture_launch(graph)
        else:
            if self.gpu_rigid:
                self.rigid_solver.validate_capture_bindings(state_in, state_out, self.control, dt)
                self.rigid_solver.mj_model.opt.timestep = dt / self.rigid_solver.rigid_substeps
            else:
                import mujoco

                solver = self.rigid_solver
                if self._native_state is None:
                    size = mujoco.mj_stateSize(solver.mj_model, mujoco.mjtState.mjSTATE_INTEGRATION)
                    self._native_state = np.empty(size)
                mujoco.mj_getState(
                    solver.mj_model, solver.mj_data, self._native_state, mujoco.mjtState.mjSTATE_INTEGRATION
                )
            self._issue_interval(state_in, state_out, dt)
        # The one read of an interval: the host decides here whether the interval is accepted.
        minimum, nonfinite, overflow = self._check.numpy()
        if nonfinite or overflow:
            raise SceneFailed(
                "nonfinite rigid state"
                if nonfinite
                else "MuJoCo Warp rejected a rigid step (capacity or solver budget)"
            )
        if math.isnan(minimum) or minimum <= 0:
            raise TissueStepRejected(
                f"Unresolved FEM tetrahedron motion: nonpositive swept volume bound J={minimum:.6g}"
            )

    def _end_step(self, dt):
        graph = self._end_graphs.get(id(self.state_0))
        if graph is not None:
            wp.capture_launch(graph)
        else:
            self._issue_end_step(self.state_0, dt)

    def _begin_frame(self):
        """Refresh the tissue tangent at the positions the frame starts from, at the nominal interval."""
        wp.copy(self._frame_particle_q, self.state_0.particle_q)
        if self._tangent_graph is not None and self.solver.dt == self.sim_dt:
            self.solver.step_time.set(self.sim_dt)
            wp.capture_launch(self._tangent_graph)
        else:
            self.solver.refresh_tangent(self._frame_particle_q, self.sim_dt)

    def step(self):
        """Advance one frame; a failure is kept in ``failure_reason`` and stops the scene until it is cleared.

        An interval that the tetrahedron guard rejects is repeated from its unchanged start with
        half the interval, down to the retry range of the coupled scenes
        (``example_fem_needle_insertion.Example.simulate``); the halved interval is kept to the
        end of the frame. Intervals are counted in whole ticks, so a frame ends exactly.
        """
        if self.failure_reason is not None:
            return
        self._begin_frame()
        minimum_retry_dt = min(self.sim_dt / 256, 5.0e-6)
        # Ticks of one nominal interval: the power of two at which a halved interval reaches the retry range.
        nominal = 2 ** max(0, math.ceil(math.log2(self.sim_dt / minimum_retry_dt)))
        ticks, total, done = nominal, self.args.substeps * nominal, 0
        while done < total:
            step_dt = self.sim_dt * ticks / nominal
            wp.launch(
                _ramp_arm_targets,
                self._ramp[0].size,
                [self._ramp[0], self._ramp[1], (done + ticks) / total, self._ramp[2], self.control.joint_target_q],
                device=self.control.joint_target_q.device,
            )
            try:
                self._attempt(step_dt)
            except TissueStepRejected as error:
                self._restore_attempt()
                self.rejected_intervals += 1
                if step_dt <= minimum_retry_dt:
                    self.failed_intervals += 1
                    self.failure_reason, self.failure_kind = str(error), "tissue"
                    return
                ticks //= 2
                self.solver.refresh_tangent(self._frame_particle_q, self.sim_dt * ticks / nominal)
                continue
            except SceneFailed as error:
                self.failure_reason, self.failure_kind = str(error), "rigid"
                return
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.elapsed += step_dt
            done += ticks
            self.accepted_intervals += 1
            self.minimum_step_dt = min(self.minimum_step_dt, step_dt)
            self._end_step(step_dt)
            if self.interval_observer is not None:
                # A measurement of every interval, asked for explicitly: it reads the accepted state.
                self.interval_observer(step_dt)
        self.sim_time = self.elapsed
        self.frames += 1

    # ------------------------------------------------------------------ start-up

    def _startup_refusal(self):
        """Why this scene captures nothing at start-up, or None if it does."""
        if not self.gpu_rigid:
            return "native rigid backend"
        if not self.device.is_cuda:
            return "CPU device"
        if not wp.is_conditional_graph_supported():
            return "conditional CUDA graphs unavailable"
        if self.elapsed > 0 or self.accepted_intervals:
            return "the scene has advanced since its reset"
        return None

    def _state_arrays(self):
        """Every array that an interval, its end-of-step call or the ramp writes and a later interval reads."""
        arrays = [self.control.joint_target_q, self.solver.velocity_increment, self._tip_travel, *self.contact.state]
        for state in (self.state_0, self.state_1):
            arrays.extend(
                getattr(state, name)
                for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
            )
        if self._checkpoint is not None:
            arrays.extend(self._checkpoint.arrays)
        return [array for array in arrays if array.size]

    def _load_run_modules(self):
        """Load the kernel modules that eager code of a frame or of an observation launches, by name."""
        from ..._src.solvers.coupled import tet_step
        from ..._src.solvers.coupled.interface import tool_tissue_contact
        from ..._src.solvers.thread import tissue
        from . import _grasp_scene

        modules = [
            # Ramp of the arm targets, every interval.
            (_grasp_scene, (256,)),
            # Checks of an interval, observation of a frame.
            (__name__, (256,)),
            (tet_step, (256,)),
            # Surface of an observation and its nearest points.
            (tissue, (256,)),
            (tool_tissue_contact, (256,)),
            *self.contact.modules(),
        ]
        for module, block_dims in modules:
            for block_dim in block_dims:
                wp.load_module(module, device=self.device, block_dim=block_dim)

    def capture_graphs(self):
        """Start-up call: capture and launch, before the first frame, every graph that a frame launches.

        On the MuJoCo Warp backend with conditional CUDA graphs: the kernel modules of the tissue
        solver and of the eager code are loaded, the interval runs once as written (which makes
        the rigid step, its pools and the Krylov workspace of the tissue solver), and then the
        tangent refresh of a frame, the interval of each of the two state buffers and its
        end-of-step call are captured and launched once. These are real steps; after each of them
        every array of :meth:`_state_arrays` gets back the bits it had, so no time passes and
        nothing is accepted.

        Returns:
            True if the graphs exist. False if the scene steps as written; ``_startup_status``
            then holds the reason and nothing is checked after start-up.
        """
        reason = self._startup_refusal()
        if reason is not None:
            self._startup_status = reason
            return False
        gc.collect()
        device, dt = self.device, self.sim_dt
        solver, rigid = self.solver, self.rigid_solver
        solver.load_kernel_modules()
        self._load_run_modules()
        self.state_1.assign(self.state_0)
        arrays = self._state_arrays()
        saved = [wp.clone(array) for array in arrays]
        native_timestep = rigid.mj_model.opt.timestep

        def restore():
            for array, values in zip(arrays, saved, strict=True):
                wp.copy(array, values)

        def warm(state_in, state_out):
            # As written: what a capture may not create is created here (the rigid step and its pools,
            # the Krylov workspace of the tissue solver). The first interval of the rigid backend can
            # switch one of its options, which drops what that interval prepared: then once more.
            for _ in range(3):
                rigid.validate_capture_bindings(state_in, state_out, self.control, dt)
                if rigid.rigid_step is not None and rigid._rigid_warmed and self._startup_warmed:
                    return
                self._issue_interval(state_in, state_out, dt)
                self._issue_end_step(state_out, dt)
                wp.synchronize_device(device)
                restore()
                self._startup_warmed = True
            rigid.validate_capture_bindings(state_in, state_out, self.control, dt)
            if rigid.rigid_step is None or not rigid._rigid_warmed:
                raise RuntimeError("The rigid interval is not prepared after three intervals")

        wp.copy(self._frame_particle_q, self.state_0.particle_q)
        solver.refresh_tangent(self._frame_particle_q, dt)
        self._startup_warmed = False
        try:
            warm(self.state_0, self.state_1)
            with capture.ScopedCapture(device=device) as captured:
                solver.refresh_tangent(self._frame_particle_q, dt)
            self._tangent_graph = captured.graph
            wp.capture_launch(self._tangent_graph)
            for state_in, state_out in ((self.state_0, self.state_1), (self.state_1, self.state_0)):
                warm(state_in, state_out)
                with capture.ScopedCapture(device=device) as captured:
                    self._issue_interval(state_in, state_out, dt)
                self._graphs[id(state_in)] = captured.graph
                wp.capture_launch(captured.graph)
                if self.contact.has_end_step:
                    with capture.ScopedCapture(device=device) as captured:
                        self._issue_end_step(state_out, dt)
                    self._end_graphs[id(state_out)] = captured.graph
                    wp.capture_launch(captured.graph)
                wp.synchronize_device(device)
                restore()
        finally:
            restore()
            rigid.mj_model.opt.timestep = native_timestep
        return True

    def finish_startup(self):
        """End the start-up here: graph captures and kernel module loads after this call are late ones."""
        mark = None
        if self._startup_ready:
            mark = capture.finish_startup(self._startup_begun, self.device)
            if mark is None:
                self._startup_status = "CPU device"
        self._startup_mark = mark

    def start(self):
        """Run the start-up call at the placed start state; the owner observes once and then marks the start-up."""
        self._startup_status = "start-up not marked"
        self._startup_ready = self.capture_graphs()

    def census(self):
        """Graph captures and kernel module loads after the start-up mark: one walk of the loaded modules."""
        mark = self._startup_mark
        if mark is None:
            return {
                "capture_census_status": self._startup_status,
                "captures_after_startup": None,
                "solver_captures_after_startup": None,
                "module_loads_after_startup": None,
                "render_module_loads_after_startup": None,
            }
        late = capture.modules(mark, RENDER_MODULE_PREFIXES)
        return {
            "capture_census_status": "checked",
            **capture.counts(mark),
            "module_loads_after_startup": late["module_loads_after_startup"],
            "render_module_loads_after_startup": late["render_module_loads_after_startup"],
            **mark.startup,
            "late_sites": late["late_sites"],
            "late_modules": late["late_modules"],
            "late_render_modules": late["late_render_modules"],
        }

    # ------------------------------------------------------------------ observation

    def _measure_capsules(self, state, first):
        wp.launch(
            _capsule_depths,
            self.capsule_point_count,
            inputs=[
                self.surface.mesh.id,
                state.body_q,
                self._capsule_point_body,
                self._capsule_point_local,
                self._capsule_point_radius,
                self.surface.triangles,
                self._vertex_held,
                self.surface_reach,
                first,
            ],
            outputs=[self._capsule_depth],
            device=self.device,
        )

    def observe(self, *, keep_travel=False):
        """Read the accepted state: the only call of this scene that reads arrays back, never called by a step.

        Returns:
            ``part_depth_m``: for every capsule of the tools, the deepest point of the drawn tool
            below the deformed tissue surface [m] (positive inside, negative clear of it, None
            beyond the reach of the search); ``surface_displacement_m`` and ``dimple_depth_m``:
            the largest displacement of a surface node and its largest part into the tissue;
            ``tip_travel_m``: the largest travel of the jaw tip in one interval since the last
            observation (``keep_travel`` leaves the track of the tip running); ``tip_gap_m``: the gap
            of the tip sphere to the tissue surface at rest (negative inside it, None beyond the
            reach of the search); ``tip_crossing``: None, or for the interval since the last
            observation in which that gap changed sign from clear to inside, the travel of the tip
            in it and the gap at its start and at its end; ``body_q`` and ``joint_q``; the
            contact's own observation; the counters of intervals.
            ``capsule_depth_m``: for every capsule, how far the exact drawn capsule is inside the
            tissue at the end of the interval [m] (zero: nowhere), measured as its radius less
            the distance of its axis from the tissue surface and, where the axis is under the
            surface, on points of the capsule surface (ray-parity sign; ``_build_capsule_points``
            gives the stations and what each kind reads), independently of any contact law: ``yielding``
            where the nearest triangle can move, ``held`` where it has a fixed vertex or one
            outside the block, ``centre`` for the centre line, and ``drawn`` for the state as it
            is drawn, after a contact has moved tissue nodes of the accepted state.
            ``drawn_part_depth_m``: for every drawn part of the tools (the CAD that the viewer
            draws), how far its deepest vertex is inside the tissue surface as it is drawn [m]
            (zero: no vertex is inside), ``yielding`` and ``held`` as above.
            ``actuator_demand_over_limit``: for every joint, the force that its position
            actuator asks for over its force limit (1 or more: the actuator is at its limit).
            ``particle_q``: with a pinch (the swept contact), the positions of the tissue nodes;
            None otherwise.
        """
        state, device = self.state_0, self.device
        if not self.contact.keeps_surface:
            self.surface.update(state.particle_q)
        force, gap, face, weights, normal, point, count = self._sample_scratch
        wp.launch(
            _begin_observation,
            self.model.body_count,
            [self._part_depth, self._displacement, self._capsule_depth],
            device=device,
        )
        self._measure_capsules(state, 0)
        wp.launch(
            _find,
            len(self.sample_bodies),
            inputs=[
                self.surface.mesh.id,
                state.body_q,
                self._sample_body,
                self._sample_local,
                self._sample_radius,
                self.surface_reach,
                force,
            ],
            outputs=[gap, face, weights, normal, point, count],
            device=device,
        )
        wp.launch(
            _part_depths, len(self.sample_bodies), [gap, face, self._sample_body, self._part_depth], device=device
        )
        # The gap of the tip sphere to the surface at rest, into the track of the tip that is read below.
        wp.launch(
            _tip_gap,
            1,
            [
                state.body_q,
                self.tip_body,
                wp.vec3(*self.tip_local),
                self.tip_radius,
                self.rest_surface.id,
                self.surface_reach,
                self._tip_travel,
            ],
            device=device,
        )
        wp.launch(
            _surface_displacement,
            len(self.surface_nodes),
            [state.particle_q, self._rest, self._surface_nodes, self._surface_normals, self._displacement],
            device=device,
        )
        if self.contact.moves_tissue:
            # The state as it is drawn: the contact has moved nodes of the accepted state. Its next
            # end-of-step call refits the surface again.
            self.surface.update(state.particle_q)
            self._measure_capsules(state, 3)
        # The drawn parts against the tissue surface as it is drawn.
        wp.launch(
            _capsule_depths,
            self.drawn_point_count,
            inputs=[
                self.surface.mesh.id,
                state.body_q,
                self._drawn_point_body,
                self._drawn_point_local,
                self._drawn_point_radius,
                self.surface.triangles,
                self._vertex_held,
                self.surface_reach,
                6,
            ],
            outputs=[self._capsule_depth],
            device=device,
        )
        depths = self._part_depth.numpy()
        displacement = self._displacement.numpy()
        capsule = self._capsule_depth.numpy()
        if not self.contact.moves_tissue:
            capsule[:, 3:6] = capsule[:, :3]
        travel, tip_gap, crossed, gap_before, gap_after = (float(value) for value in self._tip_travel.numpy())
        if not keep_travel:
            self._tip_travel.zero_()
        body_q = state.body_q.numpy()
        joint_q, joint_qd = state.joint_q.numpy(), state.joint_qd.numpy()
        demand = self._actuator_ke * (self.control.joint_target_q.numpy() - joint_q) - self._actuator_kd * joint_qd
        return {
            "capsule_depth_m": {
                name: {
                    "yielding": float(capsule[body, 0]),
                    "held": float(capsule[body, 1]),
                    "centre": float(capsule[body, 2]),
                    "drawn": float(capsule[body, 3:5].max()),
                    "drawn_centre": float(capsule[body, 5]),
                }
                for body, name in self.part_names.items()
            },
            "drawn_part_depth_m": {
                name: {"yielding": float(capsule[body, 6]), "held": float(capsule[body, 7])}
                for body, name in self.drawn_part_names.items()
            },
            "actuator_demand_over_limit": (np.abs(demand) / self._actuator_limit).tolist(),
            "joint_qd": joint_qd,
            "time_s": self.elapsed,
            "part_depth_m": {
                name: (float(depths[body]) if np.isfinite(depths[body]) else None)
                for body, name in self.part_names.items()
            },
            "surface_displacement_m": float(displacement[0]),
            "dimple_depth_m": float(displacement[1]),
            "tip_travel_m": travel,
            "tip_gap_m": tip_gap if math.isfinite(tip_gap) else None,
            "tip_crossing": {"travel_m": crossed, "gap_before_m": gap_before, "gap_after_m": gap_after}
            if crossed > 0.0
            else None,
            "body_q": body_q,
            "joint_q": joint_q,
            # The tissue nodes, for what the measurements of the pinch ask of them (a pinch exists with the swept law).
            "particle_q": state.particle_q.numpy()
            if getattr(getattr(self.contact, "exchange", None), "pinch", None) is not None
            else None,
            "contact": self.contact.observe(),
            "accepted_intervals": self.accepted_intervals,
            "rejected_intervals": self.rejected_intervals,
            "failed_intervals": self.failed_intervals,
            "failure_reason": self.failure_reason,
        }

    def tissue_integrity(self):
        """Host check of the accepted tissue state for a self-test: finite, no inverted tetrahedron, held nodes at rest."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        rest = np.asarray(self.selection.vertices)
        tetrahedra = np.asarray(self.selection.indices)

        def volumes(points):
            vertices = points[tetrahedra].astype(np.float64)
            return np.linalg.det(vertices[:, 1:] - vertices[:, :1])

        finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
        ratio = volumes(positions) / volumes(rest) if finite else np.array([np.nan])
        fixed = self.fixed_vertex_indices
        return {
            "finite": finite,
            "minimum_volume_ratio": float(ratio.min()),
            "held_nodes_at_rest": bool(finite and np.array_equal(positions[fixed], rest[fixed].astype(np.float32))),
        }

    def describe(self):
        """Host values of the scene for a report; nothing is read from the device."""
        args = self.args
        return {
            "rigid_backend": "mujoco-warp" if self.gpu_rigid else "mujoco-native",
            "device": str(self.device),
            "contact": self.contact_name,
            "tools": "PSM1 driven by arm targets and a jaw opening, PSM2 parked as in the grasp scene",
            "rigid_bodies": self.model.body_count,
            "rigid_substeps": self.rigid_solver.rigid_substeps,
            "frame_dt_s": self.frame_dt,
            "interval_dt_s": self.sim_dt,
            "tissue_region": list(self.selected_region_ids),
            "tissue_particles": self.model.particle_count,
            "tissue_tetrahedra": len(self.selection.indices),
            "held_particles": len(self.fixed_vertex_indices),
            "mesh_spacing_m": self.mesh_spacing,
            "young_modulus_Pa": args.young_modulus,
            "poisson_ratio": args.poisson_ratio,
            "density_kg_per_m3": args.density,
            "fem_iteration_cap": args.solver_iterations,
            "fem_relative_tolerance": args.fem_tolerance,
            "capsule_radii_m": [float(radius) for radius in self.capsule_radii],
            "capsule_parts": [name for tool in self.graspers for name in tool.capsule_parts],
            "capsule_bounds_m": [float(bound) for tool in self.graspers for bound in tool.capsule_bounds],
            "capsule_source": "the capsule set of the tools, fitted to the drawn tool (dvrk_tool_capsules); the rigid "
            "collision shapes, the tool/tissue contact and the view use the same rows",
            "contact_spheres": len(self.sample_bodies),
            "jaw_contact_radius_m": self.jaw_contact_radius,
            "actuator_time_constants_s": [float(value) for value in self.actuator_time_constants],
            "contact_parameters": self.contact.parameters(),
        }

    def close(self):
        """Nothing outside this process is held."""

    def geometry(self):
        """Static geometry for a view of this scene (the keys of the grasp scene's geometry that it has)."""
        reference, selection = self.reference, self.selection
        return {
            "tissue_rest_positions": np.asarray(selection.vertices).copy(),
            "tissue_triangles": np.asarray(selection.exterior_surface_triangles).copy(),
            "reference_positions": np.asarray(reference.vertices).copy(),
            "reference_triangles": np.asarray(reference.exterior_surface_triangles).copy(),
            "active_reference_indices": np.searchsorted(
                reference.global_vertex_indices, selection.global_vertex_indices
            ),
            "rigid_body_labels": list(self.model.body_label),
            "grippers": [
                {
                    "instrument": f"PSM{index + 1}",
                    "part_body_indices": dict(tool.part_body_indices),
                    "trocar_pose": np.asarray(tool.trocar_pose).copy(),
                    "capsule_bodies": list(tool.capsule_bodies),
                    "capsule_endpoints": tool.capsule_endpoints.copy(),
                    "capsule_radii": tool.capsule_radii.copy(),
                    "capsule_parts": list(tool.capsule_parts),
                    "parameters": vars(tool.config),
                }
                for index, tool in enumerate(self.graspers)
            ],
        }
