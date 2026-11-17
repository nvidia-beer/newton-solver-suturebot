# SPDX-License-Identifier: Apache-2.0
"""Display of the tool/tissue example: the tools' CAD and the pad surface on a render model of their own.

The parts are those the grasp example draws (original tool CAD on the physical bodies, the
complete pad exterior with the simulated block following its particles, the collision capsules
as shapes that the viewer's "Show Collision" option draws, trocar rings and the support), without
needle and thread. A frame is brought to the render model on the device: no state is read back.
With a pinch (the swept contact), the tissue nodes that a jaw carries are drawn as points:
carried nodes in one colour, those that were found clamped in another.
"""

import math

import newton
import numpy as np
import warp as wp

from ..phantom.example_fem_phantom_regions import PAD_COLOR
from ._grasp_render import _trocar_marker_mesh, collision_capsules
from ._grasp_surface import FullSurfaceGeometry
from ._grasp_view import fit_inspection_camera
from ._grasp_visual_assets import load_dvrk_tool, physical_tool_parts

# Radius of the point drawn on a tissue node that a jaw carries, in mesh spacings: a choice of the
# picture, small enough that the points of neighbouring nodes stay apart.
CARRIED_POINT_SPACINGS = 0.12
# Colours of a carried node and of one that was found clamped (RGB).
CARRIED_COLOR, CLAMPED_COLOR = (1.0, 0.82, 0.10), (0.10, 0.85, 0.25)


@wp.kernel(enable_backward=False)
def _scatter_surface(
    particle_q: wp.array[wp.vec3], particles: wp.array[int], vertices: wp.array[int], surface: wp.array[wp.vec3]
):
    # Surface vertices of the simulated block take the positions of their particles.
    index = wp.tid()
    surface[vertices[index]] = particle_q[particles[index]]


@wp.kernel(enable_backward=False)
def _carried_points(
    particle_q: wp.array[wp.vec3],
    nodes: wp.array[int],
    jaw: wp.array[int],
    clamped: wp.array[int],
    radius: float,
    carried_color: wp.vec3,
    clamped_color: wp.vec3,
    points: wp.array[wp.vec3],
    radii: wp.array[float],
    colors: wp.array[wp.vec3],
):
    # One point per free surface node: drawn with a radius where a jaw carries the node, with none elsewhere.
    index = wp.tid()
    particle = nodes[index]
    points[index] = particle_q[particle]
    if jaw[particle] < 0:
        radii[index] = 0.0
        return
    radii[index] = radius
    if clamped[particle] != 0:
        colors[index] = clamped_color
    else:
        colors[index] = carried_color


class ToolTissueView:
    """Render model of a ``ToolTissueScene`` and its camera.

    Args:
        scene: The physical scene; its bodies keep their indices in the render model.
        viewer: The viewer that draws the model.
    """

    def __init__(self, scene, viewer):
        geometry = scene.geometry()
        device = scene.device
        self.scene, self.viewer = scene, viewer
        self.surface = FullSurfaceGeometry(geometry)
        body_count = len(geometry["rigid_body_labels"])
        assets = [
            load_dvrk_tool(
                instrument=tool["instrument"], jaw_opening=0, shaft_length_m=tool["parameters"]["shaft_length_m"]
            )
            for tool in geometry["grippers"]
        ]
        builder = newton.ModelBuilder(up_axis="Z")
        cfg = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
        # This model is drawn, never stepped: the collision flag only tells the viewer what these are.
        collision_cfg = newton.ModelBuilder.ShapeConfig(
            density=0, has_shape_collision=True, has_particle_collision=False, is_visible=False
        )
        rest = self.surface.rest_positions
        builder.begin_world()
        builder.add_particles(rest, np.zeros_like(rest), np.zeros(len(rest)))
        builder.add_triangles(*self.surface.triangles.T, color=PAD_COLOR)
        bodies = [builder.add_body(is_kinematic=True, label=f"render_{index}") for index in range(body_count)]
        for asset, tool in zip(assets, geometry["grippers"], strict=True):
            for body, mesh in physical_tool_parts(asset, tool, body_count=body_count):
                builder.add_shape_mesh(bodies[body], mesh=mesh.to_newton_mesh(), cfg=cfg)
        for body, start, end, radius in collision_capsules(geometry):
            axis = end - start
            builder.add_shape_capsule(
                bodies[body],
                xform=wp.transform(
                    wp.vec3(*((start + end) / 2)), wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*axis))
                ),
                radius=radius,
                half_height=float(np.linalg.norm(axis) / 2),
                cfg=collision_cfg,
            )
        trocar_mesh = _trocar_marker_mesh()
        for index, tool in enumerate(geometry["grippers"]):
            builder.add_shape_mesh(
                -1,
                xform=wp.transform(*np.asarray(tool["trocar_pose"], dtype=float)),
                mesh=trocar_mesh,
                cfg=cfg,
                label=f"trocar_marker_{index}",
            )
        low, high = rest.min(0), rest.max(0)
        support = (low + high) / 2
        support[2] = low[2] - 0.002
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(support), wp.quat_identity()),
            hx=0.20,
            hy=0.20,
            hz=0.001,
            cfg=cfg,
            color=(0.16, 0.34, 0.37),
        )
        builder.end_world()
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        if self.model.body_count != scene.model.body_count:
            raise ValueError("The render model must hold one body per physical body")
        self._particles = wp.array(self.surface.active_particle_indices, dtype=int, device=device)
        self._vertices = wp.array(self.surface.active_surface_indices, dtype=int, device=device)
        # The carried nodes of the pinch, as points (fixed storage: one per free surface node).
        exchange = getattr(scene.contact, "exchange", None)
        self.pinch = None if exchange is None else getattr(exchange, "pinch", None)
        if self.pinch is not None:
            count = self.pinch.law.node_count
            self._carried_radius = CARRIED_POINT_SPACINGS * scene.mesh_spacing
            self._carried = (
                wp.zeros(count, dtype=wp.vec3, device=device),
                wp.zeros(count, dtype=float, device=device),
                wp.full(count, wp.vec3(*CARRIED_COLOR), dtype=wp.vec3, device=device),
            )
        viewer.set_model(self.model)
        viewer.picking_enabled = False
        viewer.show_particles = False

    def frame(self, points):
        """Point the camera at ``points`` [m], from the side of the grasp example's view."""
        if not hasattr(self.viewer, "camera"):
            return
        self.viewer.camera.fov = math.degrees(2 * math.atan(15.2908 / 24.0))
        eye, target = fit_inspection_camera(np.asarray(points, dtype=float), view_direction=(0.0, -1.0, 0.65))
        self.viewer.set_camera(wp.vec3(eye))
        self.viewer.camera.look_at(target)

    def update(self, state):
        """Bring the accepted physical ``state`` to the render model; launches and copies on the device only."""
        wp.launch(
            _scatter_surface,
            len(self._particles),
            [state.particle_q, self._particles, self._vertices, self.state.particle_q],
            device=self.model.device,
        )
        wp.copy(self.state.body_q, state.body_q)
        if self.pinch is not None:
            wp.launch(
                _carried_points,
                self.pinch.law.node_count,
                inputs=[
                    state.particle_q,
                    self.pinch.law._nodes,
                    self.pinch.jaw,
                    self.pinch.clamped,
                    self._carried_radius,
                    wp.vec3(*CARRIED_COLOR),
                    wp.vec3(*CLAMPED_COLOR),
                ],
                outputs=list(self._carried),
                device=self.model.device,
            )

    def draw(self, time):
        self.viewer.begin_frame(time)
        self.viewer.log_state(self.state)
        if self.pinch is not None:
            points, radii, colors = self._carried
            self.viewer.log_points("pinch/carried_nodes", points, radii=radii, colors=colors)
        self.viewer.end_frame()
