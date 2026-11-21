# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authored needle geometry, material map picks and a fixed-curvature insertion plan.

Planning is a reference-shape geometric check. It neither creates a tissue
passage nor predicts where a dynamically loaded needle will actually travel.
"""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from newton_solver_suturebot import NeedleGeometry


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length < 1.0e-12:
        raise ValueError("A finite nonzero direction is required")
    return vector / length


@dataclass(frozen=True)
class MaterialPoint:
    """A surface selection stored in the active FEM mesh's material coordinates."""

    region_id: int
    node_indices: np.ndarray
    weights: np.ndarray
    reference_position: np.ndarray
    normal: np.ndarray

    def position(self, positions):
        if hasattr(positions, "numpy"):
            positions = positions.numpy()
        return self.weights @ np.asarray(positions)[self.node_indices]


def pick_material_point(region_map, reference, selection, pixel_x, pixel_y):
    """Pick the exact frontmost reference surface and retain active material weights.

    Raster ownership and continuous ray ownership must agree; edge pixels with
    conflicting owners are rejected instead of moving the user's selection.
    """
    if len(selection.regions) != 1:
        raise ValueError("Select one active FEM region before selecting points")
    region_id = selection.regions[0].id
    visible = region_map.region_at(pixel_x, pixel_y)
    if visible is None:
        raise ValueError("The selected map point is background")
    if visible != region_id:
        raise ValueError("Entry and exit must belong to the selected region")
    xy = region_map.pixel_to_world_xy(pixel_x, pixel_y)
    faces = reference.vertices[reference.exterior_surface_triangles]
    edge1, edge2 = faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]
    relative = xy - faces[:, 0, :2]
    determinant = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    usable = np.abs(determinant) > 1.0e-20
    u, v = np.full(len(faces), -1.0), np.full(len(faces), -1.0)
    u[usable] = (relative[usable, 0] * edge2[usable, 1] - relative[usable, 1] * edge2[usable, 0]) / determinant[usable]
    v[usable] = (edge1[usable, 0] * relative[usable, 1] - edge1[usable, 1] * relative[usable, 0]) / determinant[usable]
    candidates = np.flatnonzero(usable & (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1 + 1.0e-9))
    if not len(candidates):
        raise ValueError("This map pixel has no corresponding visible material surface")
    z = faces[candidates, 0, 2] + u[candidates] * edge1[candidates, 2] + v[candidates] * edge2[candidates, 2]
    highest = z.max()
    ties = candidates[np.abs(z - highest) <= 1.0e-10]
    # Match the map's deterministic larger-ID tie ownership.
    owners = reference.exterior_surface_region_ids[ties]
    face = ties[np.argmax(owners)]
    if reference.exterior_surface_region_ids[face] != region_id:
        raise ValueError("The exact visible surface belongs to another region; select away from the boundary")
    global_nodes = reference.global_vertex_indices[reference.exterior_surface_triangles[face]]
    local_nodes = np.searchsorted(selection.global_vertex_indices, global_nodes)
    if np.any(local_nodes >= len(selection.vertices)) or not np.array_equal(
        selection.global_vertex_indices[local_nodes], global_nodes
    ):
        raise ValueError("The selected surface does not belong to the active FEM material")
    weights = np.array((1 - u[face] - v[face], u[face], v[face]))
    position = weights @ selection.vertices[local_nodes]
    normal = _unit(np.cross(edge1[face], edge2[face]))
    return MaterialPoint(region_id, local_nodes, weights, position, normal)


class TetLocator:
    """Reference/current Tet4 point location with a compact uniform AABB index.

    This CPU helper is for plan validation and small contact/event queries. It
    preserves holes because membership always requires tetrahedral barycentrics.
    """

    def __init__(self, vertices, tetrahedra):
        self.vertices = np.asarray(vertices, dtype=float)
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
        points = self.vertices[self.tetrahedra]
        self.origin = self.vertices.min(axis=0)
        self.cell_size = max(float(np.ptp(self.vertices, axis=0).max()) / 40, 1.0e-6)
        self.shape = np.floor(np.ptp(self.vertices, axis=0) / self.cell_size).astype(int) + 2
        self.base = points[:, 0]
        self.inverse = np.linalg.inv((points[:, 1:] - points[:, :1]).transpose(0, 2, 1))
        lower = np.floor((points.min(axis=1) - self.origin) / self.cell_size).astype(int)
        upper = np.floor((points.max(axis=1) - self.origin) / self.cell_size).astype(int)
        spans = upper - lower + 1
        counts = spans.prod(axis=1)
        owners = np.repeat(np.arange(len(points)), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        sx, sy = spans[owners, 0], spans[owners, 1]
        coordinates = lower[owners] + np.c_[offsets % sx, offsets // sx % sy, offsets // (sx * sy)]
        keys = self._key(coordinates)
        order = np.argsort(keys)
        self.keys, self.owners = keys[order], owners[order]

    def _key(self, coordinates):
        return coordinates[:, 0] + self.shape[0] * (coordinates[:, 1] + self.shape[1] * coordinates[:, 2])

    def locate(self, points, *, tolerance=1.0e-7):
        """Return owner IDs (-1 for void) and four barycentric weights."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cells = np.floor((points - self.origin) / self.cell_size).astype(int)
        keys = self._key(cells)
        owners = np.full(len(points), -1, dtype=np.int32)
        weights = np.zeros((len(points), 4))
        for i, (point, key) in enumerate(zip(points, keys, strict=True)):
            if np.any(cells[i] < 0) or np.any(cells[i] >= self.shape):
                continue
            candidates = self.owners[
                np.searchsorted(self.keys, key, side="left") : np.searchsorted(self.keys, key, side="right")
            ]
            uvw = np.einsum("nij,nj->ni", self.inverse[candidates], point - self.base[candidates])
            barycentric = np.c_[1 - uvw.sum(axis=1), uvw]
            inside = np.flatnonzero(np.min(barycentric, axis=1) >= -tolerance)
            if len(inside):
                # Prefer the deepest candidate on an internal face.
                pick = inside[np.argmax(np.min(barycentric[inside], axis=1))]
                owners[i], weights[i] = candidates[pick], barycentric[pick]
        return owners, weights


def _distance_to_triangles(point, triangles):
    """Small candidate query for closed triangle distance, including edges."""
    if not len(triangles):
        return np.inf
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    normal = np.cross(ab, ac)
    n2 = np.einsum("ij,ij->i", normal, normal)
    d = np.einsum("ij,ij->i", ap, normal)
    projected = point - d[:, None] * normal / np.maximum(n2[:, None], 1.0e-30)
    relative = projected - a
    aa, bb, cc = (np.einsum("ij,ij->i", x, y) for x, y in ((ab, ab), (ab, ac), (ac, ac)))
    da, db = np.einsum("ij,ij->i", relative, ab), np.einsum("ij,ij->i", relative, ac)
    determinant = aa * cc - bb * bb
    u = (da * cc - db * bb) / np.maximum(determinant, 1.0e-30)
    v = (db * aa - da * bb) / np.maximum(determinant, 1.0e-30)
    distances = np.where((u >= 0) & (v >= 0) & (u + v <= 1), d * d / np.maximum(n2, 1.0e-30), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.clip(
            np.einsum("ij,ij->i", point - start, edge) / np.maximum(np.einsum("ij,ij->i", edge, edge), 1.0e-30), 0, 1
        )
        distances = np.minimum(distances, np.sum((point - start - t[:, None] * edge) ** 2, axis=1))
    return float(np.sqrt(distances.min()))


@dataclass(frozen=True)
class InsertionPlan:
    """Desired rotation of the unchanged needle around a reference-space circle."""

    geometry: NeedleGeometry
    entry: MaterialPoint
    exit: MaterialPoint
    center: np.ndarray
    axis: np.ndarray
    radial: np.ndarray
    tangent: np.ndarray
    local_basis: np.ndarray
    exit_angle: float
    initial_angle: float
    finish_angle: float
    sample_spacing: float
    tissue_intervals: tuple = ()
    passage_boundaries: tuple = ()

    @property
    def expected_passages(self):
        """Reference-plan information for the example, never a physical constraint."""
        return len(self.tissue_intervals) if self.tissue_intervals else 1

    @property
    def path_points(self):
        return self.tip(np.linspace(self.initial_angle, self.finish_angle, 160))

    def tip(self, angle):
        angle = np.asarray(angle)
        return self.center + self.geometry.radius * (
            np.cos(angle)[..., None] * self.radial + np.sin(angle)[..., None] * self.tangent
        )

    def matrix(self, angle):
        radial = np.cos(angle) * self.radial + np.sin(angle) * self.tangent
        tangent = -np.sin(angle) * self.radial + np.cos(angle) * self.tangent
        rotation = np.column_stack((radial, tangent, self.axis)) @ self.local_basis.T
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.center - rotation @ self.geometry.arc_center
        return matrix

    def pose(self, angle):
        matrix = self.matrix(angle)
        quaternion = wp.quat_from_matrix(wp.mat33(*matrix[:3, :3].astype(np.float32).reshape(-1)))
        return np.r_[matrix[:3, 3], np.asarray(quaternion)]

    @classmethod
    def create(
        cls,
        geometry,
        entry,
        exit,
        selection=None,
        reference=None,
        fixed_vertices=(),
        *,
        clearance=0.00005,
        sample_spacing=None,
    ):
        if entry.region_id != exit.region_id:
            raise ValueError("Entry and exit must belong to the same active region")
        a, b = entry.reference_position, exit.reference_position
        chord = b - a
        distance = float(np.linalg.norm(chord))
        if distance <= max(1.0e-6, geometry.thickness * 2):
            raise ValueError("Entry and exit are coincident or closer than the needle diameter")
        if distance >= 2 * geometry.radius:
            raise ValueError("The selected entry/exit chord exceeds the fixed needle diameter")
        angle = float(2 * np.arcsin(distance / (2 * geometry.radius)))
        if angle >= geometry.span:
            raise ValueError("The insertion arc exceeds the available needle arc length")
        forward = chord / distance
        up = entry.normal + exit.normal
        up -= np.dot(up, forward) * forward
        if np.linalg.norm(up) < 1.0e-8:
            up = np.array((0.0, 0.0, 1.0)) - forward[2] * forward
        up = _unit(up)
        if up[2] < 0:
            up = -up
        center = (a + b) / 2 + np.sqrt(geometry.radius**2 - (distance / 2) ** 2) * up
        radial = _unit(a - center)
        tangent = _unit((b - center) / geometry.radius - np.cos(angle) * radial)
        axis = _unit(np.cross(radial, tangent))
        if tangent @ entry.normal >= -0.05 or (np.cos(angle) * tangent - np.sin(angle) * radial) @ exit.normal <= 0.05:
            raise ValueError("The fixed-curvature path does not enter and emerge through the selected surface normals")
        local_radial = _unit(geometry.tip - geometry.arc_center)
        local_tangent = _unit(np.cross(geometry.arc_normal, local_radial))
        basis = np.column_stack((local_radial, local_tangent, geometry.arc_normal))
        padding = geometry.thickness + clearance
        approach = max(0.12, 3 * padding / (geometry.radius * max(np.sin(angle / 2), 0.1)))
        finish = angle + max(0.03, 2 * padding / (geometry.radius * max(np.sin(angle / 2), 0.1)))
        spacing = min(geometry.thickness / 2, 0.00015) if sample_spacing is None else float(sample_spacing)
        if not np.isfinite((clearance, spacing)).all() or clearance < 0 or spacing <= 0:
            raise ValueError("Plan clearance must be nonnegative and sample spacing positive")
        plan = cls(geometry, entry, exit, center, axis, radial, tangent, basis, angle, -approach, finish, spacing)
        if selection is not None:
            intervals, boundaries = plan.validate_material(selection, reference, fixed_vertices, clearance=clearance)
            plan = replace(plan, tissue_intervals=intervals, passage_boundaries=boundaries)
        return plan

    def _material_passages(self, selection, locator):
        """Intersect the exact circle with every physical exterior triangle.

        Circle/plane roots followed by triangle barycentrics enumerate even
        gaps smaller than ``sample_spacing``. Tet membership between roots
        distinguishes material from empty space; no passage is created here.
        """
        triangles = selection.vertices[selection.exterior_surface_triangles]
        a = triangles[:, 0]
        edge1, edge2 = triangles[:, 1] - a, triangles[:, 2] - a
        normal = np.cross(edge1, edge2)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        cosine = self.geometry.radius * (normal @ self.radial)
        sine = self.geometry.radius * (normal @ self.tangent)
        constant = np.einsum("ij,ij->i", normal, self.center - a)
        amplitude = np.hypot(cosine, sine)
        distance_tolerance = 1.0e-10
        angular_tolerance = distance_tolerance / self.geometry.radius
        coplanar = (amplitude < distance_tolerance) & (np.abs(constant) < distance_tolerance)
        if coplanar.any():
            arc = self.tip(np.linspace(0, self.exit_angle, 65))
            lower, upper = arc.min(axis=0) - 1.0e-6, arc.max(axis=0) + 1.0e-6
            overlap = np.all(triangles.max(axis=1) >= lower, axis=1) & np.all(triangles.min(axis=1) <= upper, axis=1)
            if np.any(coplanar & overlap):
                raise ValueError(
                    "The planned circle lies along a tissue boundary; choose points away from that ambiguous surface"
                )
        usable = (amplitude > distance_tolerance) & (np.abs(constant) <= amplitude + distance_tolerance)
        faces = np.flatnonzero(usable)
        phase = np.arctan2(sine[faces], cosine[faces])
        offset = np.arccos(np.clip(-constant[faces] / amplitude[faces], -1, 1))
        angles = np.concatenate(((phase + offset) % (2 * np.pi), (phase - offset) % (2 * np.pi)))
        faces = np.tile(faces, 2)
        within = (angles > angular_tolerance) & (angles < self.exit_angle - angular_tolerance)
        angles, faces = angles[within], faces[within]
        relative = self.tip(angles) - a[faces]
        e1, e2 = edge1[faces], edge2[faces]
        aa, ab, bb = (np.einsum("ij,ij->i", x, y) for x, y in ((e1, e1), (e1, e2), (e2, e2)))
        da, db = np.einsum("ij,ij->i", relative, e1), np.einsum("ij,ij->i", relative, e2)
        determinant = aa * bb - ab * ab
        u, v = (da * bb - db * ab) / determinant, (db * aa - da * ab) / determinant
        inside = (u >= -1.0e-8) & (v >= -1.0e-8) & (u + v <= 1 + 1.0e-8)
        angles, faces, u, v = angles[inside], faces[inside], u[inside], v[inside]
        ordered = np.argsort(angles)
        crossings = [(0.0, self.entry)]
        for index in ordered:
            angle = float(angles[index])
            if angle - crossings[-1][0] <= angular_tolerance:
                continue
            face = faces[index]
            nodes = selection.exterior_surface_triangles[face]
            weights = np.array((1 - u[index] - v[index], u[index], v[index]))
            point = MaterialPoint(
                self.entry.region_id, nodes.copy(), weights, weights @ selection.vertices[nodes], normal[face].copy()
            )
            crossings.append((angle, point))
        crossings.append((self.exit_angle, self.exit))
        bounds = np.array([item[0] for item in crossings])
        if np.any(np.diff(bounds) * self.geometry.radius < 1.0e-8):
            raise ValueError(
                "The planned path has unresolved near-tangent tissue crossings; adjust the selected points"
            )
        owners, _ = locator.locate(self.tip((bounds[:-1] + bounds[1:]) / 2), tolerance=1.0e-10)
        occupied = owners >= 0
        if not occupied[0] or not occupied[-1]:
            raise ValueError("The path does not enter tissue at the selected entry and emerge at the selected exit")
        intervals, boundaries = [], []
        for index, is_material in enumerate(occupied):
            if not is_material:
                continue
            if index and occupied[index - 1]:
                intervals[-1] = (intervals[-1][0], float(bounds[index + 1]))
                boundaries[-1] = (boundaries[-1][0], crossings[index + 1][1])
            else:
                intervals.append((float(bounds[index]), float(bounds[index + 1])))
                boundaries.append((crossings[index][1], crossings[index + 1][1]))
        return tuple(intervals), tuple(boundaries)

    def validate_material(self, selection, reference, fixed_vertices, *, clearance=0.00005):
        """Allow empty wound gaps; reject inactive tissue and fixed-boundary collisions.

        Passage counting uses analytic exterior crossings, independent of the
        finite sampling used for swept-shaft clearance. These reference queries
        never authorize actual puncture, constrain handling or create a track.
        """
        active = TetLocator(selection.vertices, selection.indices)
        intervals, boundaries = self._material_passages(selection, active)
        total = self.finish_angle - self.initial_angle + self.geometry.span
        count = max(8, int(np.ceil(total * self.geometry.radius / self.sample_spacing)))
        angles = np.linspace(self.initial_angle - self.geometry.span, self.finish_angle, count + 1)
        points = self.tip(angles)
        padding = self.geometry.thickness + clearance + self.sample_spacing / 2
        # Full tube disks include the modeled capsule radius. Bounding-box
        # candidates plus closed-triangle distance conservatively reject support contact.
        fixed_vertices = np.asarray(fixed_vertices, dtype=int)
        fixed_faces = selection.surface_triangles[np.isin(selection.surface_triangles, fixed_vertices).all(axis=1)]
        triangles = selection.vertices[fixed_faces]
        if len(triangles):
            lower, upper = triangles.min(axis=1) - padding, triangles.max(axis=1) + padding
            for point in points:
                nearby = np.all((point >= lower) & (point <= upper), axis=1)
                if _distance_to_triangles(point, triangles[nearby]) <= padding:
                    raise ValueError("The swept needle collides with a fixed pad boundary")
        if reference is not None:
            full = TetLocator(reference.vertices, reference.indices)
            owners, _ = full.locate(points)
            occupied = owners >= 0
            if np.any(reference.tetrahedron_region_ids[owners[occupied]] != self.entry.region_id):
                raise ValueError("The swept needle requires interaction with unsupported inactive material")
            inactive_faces = reference.exterior_surface_triangles[
                reference.exterior_surface_region_ids != self.entry.region_id
            ]
            inactive = reference.vertices[inactive_faces]
            if len(inactive):
                lower, upper = inactive.min(axis=1) - padding, inactive.max(axis=1) + padding
                for point in points:
                    nearby = np.all((point >= lower) & (point <= upper), axis=1)
                    if _distance_to_triangles(point, inactive[nearby]) <= padding:
                        raise ValueError("The swept needle contacts the surface of unsupported inactive material")
            # Initial clearance includes the complete capsule radius, not just tip position.
            count = int(np.ceil(self.geometry.span * self.geometry.radius / self.sample_spacing)) + 1
            needle = self.tip(np.linspace(self.initial_angle - self.geometry.span, self.initial_angle, count))
            owners, _ = full.locate(needle)
            if np.any(owners >= 0):
                raise ValueError("The available exterior approach starts with part of the needle inside the pad")
            surface = reference.vertices[reference.exterior_surface_triangles]
            lower, upper = surface.min(axis=1) - padding, surface.max(axis=1) + padding
            for point in needle:
                nearby = np.all((point >= lower) & (point <= upper), axis=1)
                if _distance_to_triangles(point, surface[nearby]) <= padding:
                    raise ValueError("The available exterior approach does not clear the pad by the needle radius")
        return intervals, boundaries


@dataclass(frozen=True)
class WoundGap:
    """A gap that the planned needle circle crosses between two passages, as four material points of the surface.

    ``walls`` are the two points where the circle crosses the walls of the gap: where it leaves
    the tissue of the passage before the gap and where it enters that of the passage after it.
    ``rims`` are the tops of those two walls (:func:`wound_gaps`). ``level`` is the unit
    horizontal direction from the planned entry to the planned exit.
    """

    walls: tuple
    rims: tuple
    level: np.ndarray

    @property
    def points(self):
        """The four material points in the order of :meth:`widths`: the two walls, then the two rims."""
        return (*self.walls, *self.rims)

    def widths(self, points):
        """Width of the gap for the positions of :attr:`points` [m], shape (4, 3): at the track and at the top.

        At the track: the distance between the two wall points, as the suture example measures a
        wound between the last track point of one lip and the first of the other. At the top: the
        distance between the two rim points along :attr:`level`, the opening seen from above.
        """
        points = np.asarray(points, dtype=float).reshape(4, 3)
        return float(np.linalg.norm(points[1] - points[0])), float(abs((points[3] - points[2]) @ self.level))

    def rest_widths(self):
        """:meth:`widths` of the reference shape."""
        return self.widths([point.reference_position for point in self.points])

    def report(self, points):
        """The widths for the positions of :attr:`points` and of the reference shape [m], by name."""
        track, top = self.widths(points)
        rest_track, rest_top = self.rest_widths()
        return {"track_m": track, "top_m": top, "rest_track_m": rest_track, "rest_top_m": rest_top}


def wound_gaps(plan, selection):
    """The gaps of a plan with more than one passage, each with material points on its walls and its rims.

    The planned circle leaves the tissue at the far end of one passage and enters it again at
    the near end of the next (``plan.passage_boundaries``): two points on the walls of the gap
    between two lips of a wound, at the depth of the track. The rim of a wall is found on the
    profile of the surface in the plane of the circle, which holds both points. A piece of
    that profile belongs to a wall while it rises at least as much as it runs (45 degrees to
    the level, Z up): the top of a lip is flatter, the wall of a gap steeper. From the wall
    point the profile is followed upwards, from triangle to triangle across their shared
    edges, while it is wall and still rises; where that ends is the rim, a point on an edge
    of the last wall triangle. A wall point on a triangle that is not steep is its own rim.

    The points are material points of the surface: they move with the tissue, and a width
    measured between them is the width of the same piece of the wound in every state.

    Args:
        plan: An :class:`InsertionPlan` made with the selection (its passages are known).
        selection: The active mesh (``vertices``, ``exterior_surface_triangles``).

    Returns:
        One :class:`WoundGap` per gap between two consecutive passages, in the order of the
        passages; empty for a plan with one passage.
    """
    boundaries = plan.passage_boundaries
    if len(boundaries) < 2:
        return ()
    vertices = np.asarray(selection.vertices, dtype=float)
    triangles = np.asarray(selection.exterior_surface_triangles)
    offsets = (vertices - plan.center) @ plan.axis
    above = offsets > 0.0
    crossed = np.flatnonzero(above[triangles].any(axis=1) & ~above[triangles].all(axis=1))
    faces_of, by_nodes = {}, {}
    for face in crossed:
        nodes = [int(node) for node in triangles[face]]
        by_nodes[tuple(sorted(nodes))] = int(face)
        for a, b in zip(nodes, nodes[1:] + nodes[:1], strict=True):
            if above[a] != above[b]:
                faces_of.setdefault((min(a, b), max(a, b)), []).append(int(face))

    def crossing(edge):
        """Where the plane crosses an edge: the point and its fraction of the way from the first node."""
        a, b = edge
        fraction = offsets[a] / (offsets[a] - offsets[b])
        return (1.0 - fraction) * vertices[a] + fraction * vertices[b], fraction

    def edges(face):
        nodes = [int(node) for node in triangles[face]]
        pairs = [(min(a, b), max(a, b)) for a, b in zip(nodes, nodes[1:] + nodes[:1], strict=True)]
        return [edge for edge in pairs if edge in faces_of]

    def steep(face):
        first, second = (crossing(edge)[0] for edge in edges(face))
        run = second - first
        return abs(run[2]) >= np.hypot(run[0], run[1])

    def rim(wall):
        face = by_nodes.get(tuple(sorted(int(node) for node in wall.node_indices)))
        if face is None or len(edges(face)) != 2 or not steep(face):
            return wall
        edge = max(edges(face), key=lambda item: crossing(item)[0][2])
        for _ in range(len(crossed)):
            beyond = [other for other in faces_of[edge] if other != face]
            if len(beyond) != 1 or len(edges(beyond[0])) != 2 or not steep(beyond[0]):
                break
            onward = next(item for item in edges(beyond[0]) if item != edge)
            if crossing(onward)[0][2] <= crossing(edge)[0][2]:
                break
            face, edge = beyond[0], onward
        nodes = triangles[face]
        point, fraction = crossing(edge)
        weights = np.zeros(3)
        weights[int(np.flatnonzero(nodes == edge[0])[0])] = 1.0 - fraction
        weights[int(np.flatnonzero(nodes == edge[1])[0])] = fraction
        corner = vertices[nodes]
        return MaterialPoint(
            wall.region_id, nodes.copy(), weights, point, _unit(np.cross(corner[1] - corner[0], corner[2] - corner[0]))
        )

    level = plan.exit.reference_position - plan.entry.reference_position
    level = _unit(np.array([level[0], level[1], 0.0]))
    gaps = []
    for before, after in zip(boundaries[:-1], boundaries[1:], strict=True):
        walls = (before[1], after[0])
        gaps.append(WoundGap(walls, tuple(rim(wall) for wall in walls), level))
    return tuple(gaps)
