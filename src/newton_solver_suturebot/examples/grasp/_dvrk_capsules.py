"""The collision capsules of a dVRK tool, fitted to the tool that is drawn.

One capsule set per tool stands for it everywhere: the rigid collision shapes of the MuJoCo
model, the tool/tissue contact, the measurements and the collision shapes that the viewer
draws all read the set that the tool model publishes (``GrasperAssembly.capsule_bodies``,
``capsule_endpoints``, ``capsule_radii``). The set is not typed in: every axis and radius
comes from the CAD mesh of the part that is drawn on the same body (:func:`dvrk_tool_capsules`).

The rule (:func:`fit_capsules`): the surface of a part is divided into cells, along the part
by planes across its axis and across it into rows side by side; each cell gets the smallest
capsule about the centre line of the cell that contains the surface of the cell
(:func:`enclosing_capsule`). A round capsule cannot follow a right-angled edge of a drawn
part: with the edge on its surface it reaches (1 - 1/sqrt(2)) r beyond both faces of the
edge. The capsule of a part with edges is therefore drawn in until the edge is as far
outside the capsule as the capsule is outside the faces: both are EDGE_SHARE * r, with
EDGE_SHARE = (sqrt(2) - 1) / (sqrt(2) + 1) = 3 - 2 sqrt(2) = 0.172. That is the bound of the
set: no point of the drawn part is further outside it than EDGE_SHARE times the radius of
the capsule of its cell (``bounds``), and nothing is outside the capsule of a round part.
:func:`capsule_cover` measures both ways on any set.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

# What a capsule and a right-angled edge of a drawn part share, as a part of the capsule's
# radius: the edge is this far outside the capsule and the capsule this far outside both faces.
EDGE_SHARE = 3.0 - 2.0 * math.sqrt(2.0)
# Smallest length of a capsule axis [m]: the remainder of the tool/tissue contact in tissue
# that yields (20 um). A cell that a ball contains gets an axis of this length; MuJoCo and
# the tool/tissue contact take capsules.
MINIMUM_AXIS_LENGTH = 2.0e-5
# Points of a drawn surface are taken no further apart than this part of the smallest
# extent of the part.
SAMPLING = 1.0 / 64.0


def _circle(points):
    """Smallest circle through one, two or three boundary points [m]: centre (2,) and radius."""
    if len(points) == 1:
        return points[0], 0.0
    if len(points) == 2:
        centre = 0.5 * (points[0] + points[1])
        return centre, float(np.linalg.norm(points[0] - centre))
    a, b, c = points
    ab, ac = b - a, c - a
    cross = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(cross) <= 1e-30:
        # Three points on a line: the two that are farthest apart.
        pairs = ((a, b), (a, c), (b, c))
        first, second = max(pairs, key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])))
        return _circle([first, second])
    ab2, ac2 = ab @ ab, ac @ ac
    centre = a + np.array([ac[1] * ab2 - ab[1] * ac2, ab[0] * ac2 - ac[0] * ab2]) / (2.0 * cross)
    return centre, float(np.linalg.norm(a - centre))


def smallest_circle(points):
    """Smallest circle that contains points of a plane.

    The circle of a few of the points is made by Welzl's construction (a point outside the
    circle of the points before it lies on the boundary of the circle of the points up to
    it), starting with the points that reach furthest in sixteen directions. While a point
    of the whole set is outside that circle, the one furthest out joins the few. The circle
    that ends this is the smallest circle of the few and contains all points, so it is the
    smallest circle of all of them.

    Args:
        points: Points [m], shape (count, 2).

    Returns:
        Centre [m], shape (2,), and radius [m].
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("The smallest circle needs finite points")
    slack = 1e-12 * max(1.0, float(np.abs(points).max()))
    angles = np.arange(16) * (np.pi / 16.0)
    reach = np.stack((np.cos(angles), np.sin(angles)), axis=1) @ points.T
    chosen = list(dict.fromkeys([*reach.argmax(axis=1).tolist(), *reach.argmin(axis=1).tolist()]))
    while True:
        centre, radius = _welzl(points[chosen], slack)
        distance = np.linalg.norm(points - centre, axis=1)
        furthest = int(distance.argmax())
        if distance[furthest] <= radius + slack:
            return np.asarray(centre, dtype=float), float(radius)
        chosen.append(furthest)


def _welzl(points, slack):
    """Smallest circle of a few points of a plane, in their order: centre and radius."""

    def outside(point, centre, radius):
        return np.linalg.norm(point - centre) > radius + slack

    centre, radius = points[0], 0.0
    for i in range(1, len(points)):
        if not outside(points[i], centre, radius):
            continue
        centre, radius = points[i], 0.0
        for j in range(i):
            if not outside(points[j], centre, radius):
                continue
            centre, radius = _circle([points[i], points[j]])
            for k in range(j):
                if outside(points[k], centre, radius):
                    centre, radius = _circle([points[i], points[j], points[k]])
    return centre, radius


def enclosing_capsule(points, direction, *, minimum_length=0.0):
    """Smallest capsule with its axis along ``direction`` that contains ``points``.

    Across the axis the points are seen as points of a plane; the axis goes through the centre
    of their smallest circle and the radius of the capsule is the radius of that circle. Along
    the axis the capsule is as short as its round caps allow: a point at the distance d from
    the axis line is inside a cap of the radius r whose centre is no further than
    sqrt(r^2 - d^2) along the axis from it.

    Args:
        points: Points [m], shape (count, 3).
        direction: Direction of the axis, shape (3,).
        minimum_length: Smallest length of the axis [m]. Where the caps alone contain the points
            (a part that is no longer than it is thick), the axis gets this length about the
            place of the ball that contains them.

    Returns:
        The two end points of the axis [m], shape (2, 3), and the radius [m].
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    first = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = first - (first @ direction) * direction
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    centre, radius = smallest_circle(np.stack((points @ first, points @ second), axis=1))
    origin = centre[0] * first + centre[1] * second
    along = points @ direction
    across = np.linalg.norm(points - origin - along[:, None] * direction, axis=1)
    reach = np.sqrt(np.maximum(radius * radius - across * across, 0.0))
    start, end = float((along + reach).min()), float((along - reach).max())
    if end - start < minimum_length:
        middle = 0.5 * (start + end)
        start, end = middle - 0.5 * minimum_length, middle + 0.5 * minimum_length
    return np.stack((origin + start * direction, origin + end * direction)), radius


def fit_capsules(
    points,
    axis,
    wide,
    *,
    boundaries=(),
    rows_wide=1,
    rows_thin=1,
    edges=False,
    margin=0.0,
    minimum_length=MINIMUM_AXIS_LENGTH,
):
    """Capsules that together contain every point of a drawn part.

    The points are divided into cells: along ``axis`` by planes at ``boundaries``, then each
    of those pieces across ``wide`` into ``rows_wide`` slabs of equal width and across the
    third direction into ``rows_thin`` slabs, between the smallest and the largest coordinate
    of the piece. A cell gets the smallest capsule about its centre line that contains its
    points (:func:`enclosing_capsule`); the centre line runs from the middle of the points in
    the first eighth of the cell's length to the middle of those in the last eighth.

    A flat part takes two or three rows side by side across its width, a tapered or curved one
    a chain of pieces along its axis, a round one a single capsule.

    Args:
        points: Vertices of the drawn part in the frame of its body [m], shape (count, 3).
        axis: Direction along the part, shape (3,).
        wide: Direction across the part in which it is widest, shape (3,).
        boundaries: Coordinates along ``axis`` [m] at which the part is cut into pieces.
        rows_wide: Capsules side by side across ``wide`` in every piece.
        rows_thin: Capsules side by side across the third direction in every piece.
        edges: Whether the part has edges that a round capsule cannot follow. Then every
            capsule is drawn in by the share of a right-angled edge: its radius is that of the
            smallest capsule that contains its cell, divided by 1 + EDGE_SHARE.
        margin: How far a cell reaches beyond the planes that bound it [m]. With the points of a
            surface that are ``margin`` apart (:func:`surface_points`), every point of the
            surface lies between three of them that belong to its own cell, so the capsule of
            the cell, which is convex, contains it: the cells then leave nothing of the surface
            out between two points.
        minimum_length: Smallest length of a capsule axis [m] (:func:`enclosing_capsule`).

    Returns:
        End points of the capsule axes [m], shape (capsules, 2, 3), radii [m], shape
        (capsules,), piece by piece from the start of the axis, row by row, and for every
        capsule the bound [m] on how far a point of its cell can be outside it: EDGE_SHARE
        times its radius with ``edges``, zero without.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    wide = np.asarray(wide, dtype=float)
    wide = wide - (wide @ axis) * axis
    wide /= np.linalg.norm(wide)
    thin = np.cross(axis, wide)
    if min(rows_wide, rows_thin) < 1 or min(minimum_length, margin) < 0.0:
        raise ValueError("A piece takes at least one row each way; margin and smallest axis length are nonnegative")
    along = points @ axis
    planes = [-math.inf, *sorted(float(value) for value in boundaries), math.inf]
    endpoints, radii, bounds = [], [], []
    for low, high in zip(planes[:-1], planes[1:], strict=True):
        piece = points[(along >= low - margin) & (along <= high + margin)]
        if not len(piece):
            raise ValueError("A piece of the part between two boundaries holds no vertex")
        cells = [piece]
        for direction, rows in ((wide, rows_wide), (thin, rows_thin)):
            split = []
            for cell in cells:
                coordinate = cell @ direction
                limits = np.linspace(coordinate.min(), coordinate.max(), rows + 1)
                split.extend(
                    cell[(coordinate >= first - margin) & (coordinate <= second + margin)]
                    for first, second in zip(limits[:-1], limits[1:], strict=True)
                )
            cells = split
        for cell in cells:
            if not len(cell):
                raise ValueError("A row of the part holds no vertex")
            coordinate = cell @ axis
            length = float(np.ptp(coordinate))
            near = cell[coordinate <= coordinate.min() + length / 8.0]
            far = cell[coordinate >= coordinate.max() - length / 8.0]
            line = 0.5 * (far.min(axis=0) + far.max(axis=0)) - 0.5 * (near.min(axis=0) + near.max(axis=0))
            if np.linalg.norm(line) <= 1e-9 * max(1.0, length) or line @ axis <= 0.0:
                line = axis
            ends, radius = enclosing_capsule(cell, line, minimum_length=minimum_length)
            drawn_in = radius / (1.0 + EDGE_SHARE) if edges else radius
            endpoints.append(ends)
            radii.append(drawn_in)
            bounds.append(radius - drawn_in)
    return np.asarray(endpoints), np.asarray(radii), np.asarray(bounds)


def distance_to_capsules(points, endpoints, radii):
    """Signed distance of points from the surface of a set of capsules [m]: negative inside one of them."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    result = np.full(len(points), np.inf)
    for (start, end), radius in zip(np.asarray(endpoints, dtype=float), radii, strict=True):
        line = end - start
        fraction = np.clip((points - start) @ line / (line @ line), 0.0, 1.0)
        result = np.minimum(result, np.linalg.norm(points - start - fraction[:, None] * line, axis=1) - radius)
    return result


def capsule_surface_points(endpoints, radii, *, spacing):
    """Points on the surfaces of capsules, no further apart than ``spacing`` [m]: rings around every axis and on both caps."""
    result = []
    for (start, end), radius in zip(np.asarray(endpoints, dtype=float), radii, strict=True):
        line = end - start
        length = float(np.linalg.norm(line))
        direction = line / length
        first = np.eye(3)[int(np.argmin(np.abs(direction)))]
        first = first - (first @ direction) * direction
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        angles = np.linspace(0.0, 2.0 * np.pi, max(8, math.ceil(2.0 * np.pi * radius / spacing)), endpoint=False)
        ring = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
        for fraction in np.linspace(0.0, 1.0, math.ceil(length / spacing) + 1):
            result.append(start + fraction * line + radius * ring)
        polar = np.linspace(0.0, 0.5 * np.pi, math.ceil(0.5 * np.pi * radius / spacing) + 1)[1:]
        for tip, sign in ((start, -1.0), (end, 1.0)):
            for angle in polar:
                result.append(tip + radius * (np.cos(angle) * ring + np.sin(angle) * sign * direction))
    return np.concatenate(result)


def surface_points(vertices, triangles, *, spacing):
    """Points of a triangle surface no further apart than ``spacing`` [m]: its vertices and points of its triangles.

    A triangle whose longest edge is n spacings long gets the points whose weights on its
    three corners are multiples of 1 / n.
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    corners = vertices[np.asarray(triangles).reshape(-1, 3)]
    counts = np.ceil(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1) / spacing).astype(int)
    samples = [vertices]
    for count in np.unique(counts):
        if count < 2:
            continue
        weights = np.asarray(
            [(i, j, count - i - j) for i in range(count + 1) for j in range(count + 1 - i)], dtype=float
        )
        samples.append(np.einsum("wc,tcd->twd", weights / count, corners[counts == count]).reshape(-1, 3))
    return np.concatenate(samples)


def capsule_cover(vertices, triangles, endpoints, radii, *, spacing, drawn=None):
    """How well a set of capsules stands for a drawn part: what sticks out of what [m].

    Args:
        vertices: Vertices of the drawn part [m], shape (count, 3).
        triangles: Its triangles, shape (triangles, 3).
        endpoints: End points of the capsule axes [m], shape (capsules, 2, 3).
        radii: Capsule radii [m].
        spacing: Largest distance between two points at which either surface is sampled [m].
        drawn: Vertices and triangles of everything that is drawn around the part (the whole
            tool), which the capsules are held against; ``None`` takes the part itself.

    Returns:
        A dictionary with ``drawn_outside_m``, the largest distance of a point of the drawn
        part outside the set (zero or less: all of it is inside), that point
        (``drawn_outside_at``), ``capsules_outside_m``, for every capsule the largest distance
        of a point of its surface that is on the outside of the set from the drawn surface
        (zero for a capsule that is wholly inside the others), and those points
        (``capsules_outside_at``).
    """
    points = surface_points(vertices, triangles, spacing=spacing)
    outside = distance_to_capsules(points, endpoints, radii)
    mesh_vertices, mesh_triangles = (vertices, triangles) if drawn is None else drawn
    endpoints, radii = np.asarray(endpoints, dtype=float), np.asarray(radii, dtype=float)
    away, where = [], []
    for capsule in range(len(radii)):
        surface = capsule_surface_points(
            endpoints[capsule : capsule + 1], radii[capsule : capsule + 1], spacing=spacing
        )
        others = np.arange(len(radii)) != capsule
        if others.any():
            surface = surface[distance_to_capsules(surface, endpoints[others], radii[others]) > 0.0]
        if not len(surface):
            away.append(0.0)
            where.append(np.full(3, np.nan))
            continue
        distance = _distance_to_mesh(surface, mesh_vertices, mesh_triangles)
        away.append(float(distance.max()))
        where.append(surface[int(distance.argmax())])
    return {
        "drawn_outside_m": float(outside.max()),
        "drawn_outside_at": points[int(outside.argmax())],
        "capsules_outside_m": np.asarray(away),
        "capsules_outside_at": np.asarray(where),
    }


def _distance_to_mesh(points, vertices, triangles):
    """Distance of points from a triangle surface [m] (Warp's closest-point query on the CPU)."""
    import warp as wp

    with wp.ScopedDevice("cpu"):
        mesh = wp.Mesh(
            wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(triangles, dtype=np.int32).reshape(-1), dtype=int),
        )
        queries = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3)
        result = wp.zeros(len(points), dtype=float)
        wp.launch(_nearest_kernel(), len(points), inputs=[mesh.id, queries], outputs=[result])
        return result.numpy().astype(float)


@functools.cache
def _nearest_kernel():
    """The closest-point kernel, made when a cover is first measured: building a tool model loads no kernel."""
    import warp as wp

    @wp.kernel(enable_backward=False, module="unique")
    def nearest(mesh: wp.uint64, points: wp.array[wp.vec3], distance: wp.array[float]):
        index = wp.tid()
        query = wp.mesh_query_point_no_sign(mesh, points[index], 1.0e3)
        if query.result:
            distance[index] = wp.length(wp.mesh_eval_position(mesh, query.face, query.u, query.v) - points[index])
        else:
            distance[index] = wp.inf

    return nearest


@dataclass(frozen=True)
class ToolCapsules:
    """The capsule set of one dVRK tool, every capsule in the frame of the part it stands for.

    Attributes:
        parts: Drawn part of each capsule: "roll" (the shaft), "pitch" (the wrist), "jaw1" or "jaw2".
        endpoints: End points of the capsule axes in the frames of their parts [m], shape (capsules, 2, 3).
        radii: Capsule radii [m], shape (capsules,).
        bounds: For each capsule, how far a point of the drawn surface of its cell can be outside it [m].
        frames: For each drawn part, its pose in the wrist-yaw frame at zero wrist and jaw angles (4 x 4).
        root_ends: For each jaw, where its root ends and its blade begins, along the jaw from its hinge [m].
    """

    parts: tuple[str, ...]
    endpoints: np.ndarray
    radii: np.ndarray
    bounds: np.ndarray
    frames: dict
    root_ends: dict

    def of(self, part):
        """End points [m] and radii [m] of the capsules of one drawn part, in the frame of that part."""
        chosen = np.asarray([name == part for name in self.parts])
        return self.endpoints[chosen].copy(), self.radii[chosen].copy()


# Along which axis of its own frame each drawn part of the tool lies, and across which it is widest.
# The shaft and the wrist are bodies of revolution or nearly square: one capsule. A jaw is flat:
# two capsules side by side, across its closing direction at its root and across its hinge axis
# along its blade, and tapered: its blade is a chain of three pieces.
_ROLL_AXIS, _PITCH_AXIS, _JAW_AXIS = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
_JAW_CLOSING, _JAW_HINGE = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
JAW_ROOT_ROWS, JAW_BLADE_ROWS, JAW_BLADE_PIECES = 2, 2, 3


def jaw_capsules(vertices, triangles):
    """Capsules of one drawn jaw in the frame of its body: hinge axis z through the origin, the jaw along +y.

    The root of the jaw is its hub about the hinge and what is as thick as the hub: it reaches
    to the last section across the jaw that is thicker across the closing direction (x) than
    the radius of the hub, which is how far the jaw reaches behind its hinge. The root is flat
    across the hinge axis and gets ``JAW_ROOT_ROWS`` capsules side by side across the closing
    direction. The blade beyond it is flat across the closing direction and tapered: a chain
    of ``JAW_BLADE_PIECES`` pieces of equal length, each ``JAW_BLADE_ROWS`` capsules side by
    side across the hinge axis. The jaw has edges (:func:`fit_capsules`).

    Returns:
        End points [m], radii [m] and bounds [m] of the capsules, the root first, and where the
        root ends along the jaw [m].
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    step = SAMPLING * float(np.ptp(vertices, axis=0).min())
    points = surface_points(vertices, triangles, spacing=step)
    along = points[:, 1]
    behind = vertices[vertices[:, 1] <= 0.0]
    if not len(behind):
        raise ValueError("A jaw needs a hub about its hinge axis")
    hub = float(np.linalg.norm(behind[:, :2], axis=1).max())
    # Sections across the jaw, one sampling distance thick.
    section = np.floor(along / step).astype(np.int64)
    order = np.argsort(section, kind="stable")
    starts = np.flatnonzero(np.diff(section[order], prepend=section[order][0] - 1))
    closing = points[order, 0]
    thickness = np.maximum.reduceat(closing, starts) - np.minimum.reduceat(closing, starts)
    thick = section[order][starts][thickness > hub]
    if not len(thick):
        raise ValueError("The root of a jaw must be thicker across its closing direction than the radius of its hub")
    root_end = float((thick.max() + 1) * step)
    tip = float(along.max())
    if root_end >= tip:
        raise ValueError("A jaw needs a blade beyond its root")
    # The blade begins where the root ends. The root reaches two sampling distances into the blade: a point of
    # the surface in the first of them lies between points of the root's set, one further on between points of
    # the blade's (see ``margin`` of fit_capsules), so nothing is left out between root and blade either.
    root = fit_capsules(
        points[along <= root_end + 2.0 * step],
        _JAW_AXIS,
        _JAW_CLOSING,
        rows_wide=JAW_ROOT_ROWS,
        edges=True,
        margin=step,
    )
    pieces = np.linspace(root_end, tip, JAW_BLADE_PIECES + 1)[1:-1]
    blade = fit_capsules(
        points[along >= root_end],
        _JAW_AXIS,
        _JAW_HINGE,
        boundaries=pieces,
        rows_wide=JAW_BLADE_ROWS,
        edges=True,
        margin=step,
    )
    return (*(np.concatenate((first, second)) for first, second in zip(root, blade, strict=True)), root_end)


def _part_frame(asset, mesh):
    """Vertices of a drawn part in its own frame [m], its triangles and its pose in the wrist-yaw frame."""
    frame = np.asarray(asset.component_transforms[mesh.name], dtype=float)
    # The loader assembles the tool in its wrist-yaw frame: back to the frame of the part.
    vertices = (np.asarray(mesh.vertices, dtype=float) - frame[:3, 3]) @ frame[:3, :3]
    return vertices, np.asarray(mesh.triangles).reshape(-1, 3), frame


@functools.lru_cache(maxsize=4)
def _distal_capsules(instrument):
    """Capsules of the wrist and of both jaws of an instrument: they do not depend on the length of the shaft."""
    from ._grasp_visual_assets import load_dvrk_tool

    asset = load_dvrk_tool(instrument=instrument, jaw_opening=0)
    result, frames, root_ends = {}, {}, {}
    for mesh in asset.meshes:
        vertices, triangles, frames[mesh.name] = _part_frame(asset, mesh)
        if mesh.name in ("jaw1", "jaw2"):
            *result[mesh.name], root_ends[mesh.name] = jaw_capsules(vertices, triangles)
        elif mesh.name == "pitch":
            # One capsule: it contains the surface if it contains the vertices.
            result[mesh.name] = fit_capsules(vertices, _PITCH_AXIS, (0.0, 1.0, 0.0), edges=True)
    return result, frames, root_ends


@functools.lru_cache(maxsize=8)
def dvrk_tool_capsules(instrument="PSM1", shaft_length_m=None):
    """The capsule set of a dVRK tool, fitted to the CAD that is drawn (``load_dvrk_tool``).

    The shaft (round) is one capsule that contains it. The wrist (square in section, with
    edges) is one capsule; each jaw is the set of :func:`jaw_capsules`. The pin of the
    wrist-yaw joint lies inside the roots of the jaws and gets none. The result is kept: a
    tool model of the same instrument and shaft length gets the same set.

    Args:
        instrument: "PSM1" (large needle driver) or "PSM2" (DeBakey forceps).
        shaft_length_m: Centreline length of the shaft [m] of the tool model; ``None`` takes the source CAD.

    Returns:
        The :class:`ToolCapsules` of the tool, the shaft first, then the wrist and the two jaws.
    """
    from ._grasp_visual_assets import load_dvrk_tool

    asset = load_dvrk_tool(instrument=instrument, jaw_opening=0, shaft_length_m=shaft_length_m)
    distal, frames, root_ends = _distal_capsules(instrument)
    roll = next(mesh for mesh in asset.meshes if mesh.name == "roll")
    vertices, _, _ = _part_frame(asset, roll)
    fitted = {"roll": fit_capsules(vertices, _ROLL_AXIS, (0.0, 1.0, 0.0)), **distal}
    names = ("roll", "pitch", "jaw1", "jaw2")
    result = ToolCapsules(
        tuple(name for name in names for _ in fitted[name][1]),
        *(np.concatenate([fitted[name][index] for name in names]) for index in range(3)),
        dict(frames),
        dict(root_ends),
    )
    for array in (result.endpoints, result.radii, result.bounds):
        array.setflags(write=False)
    return result
