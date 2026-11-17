# SPDX-License-Identifier: Apache-2.0
"""Scene content of the surgical incision examples: the Baymax body, its size, its place and its parts.

Data and the loader only; nothing here simulates. The asset is read as it is stored
(``assets/baymax/README.md``); scale and placement happen here, at load. Every number below is
either a fact of the asset, measured on the file and measured again by
``tests/test_baymax_asset.py``, or derived from such facts where it stands, or named a choice
with its reason.

Frames. The asset's own unit is its span from hand to hand, 1.000 along x; the head is at +y,
the feet at -y, the belly at +z, the back at -z. The scene frame is the frame of the SutureBot
pad [m], in which the dVRK ports of the recordings are placed: the body lies on its back, belly
up, arm span along x.
"""

import math
from pathlib import Path

import numpy as np

from newton_solver_suturebot import load_mesh

ASSET = Path(__file__).parents[1] / "assets/baymax/tet_baymax.usda"
SHA256 = "c989ec2eb70d88ff2739c209720354e528440d54a4e353f25c0be3cd469876c4"
POINTS, TETRAHEDRA, SURFACE_TRIANGLES = 8965, 38730, 9882

# ---------------------------------------------------------------------------- facts of the asset
# In asset units (span = 1). Measured on the file.
# Size of the bounding box: arm span, head to feet, back to belly.
EXTENT = (1.0, 0.632466, 0.326532)
# Mean length of the three edges that leave the first corner of each tetrahedron: the tissue edge
# as the suture scenes measure it (_src/solvers/thread/worker_thread.py, ``tissue_edge_m``).
MEAN_EDGE = 0.019971
# Median edge of the surface triangles (the belly's own median is 0.013997).
SURFACE_EDGE = 0.013952
# Smallest altitude of a tetrahedron.
SMALLEST_ALTITUDE = 0.00091258
# The highest node of the belly, node 2459: the belly's apex.
BELLY_APEX = (0.000988, -0.040272, 0.163266)

# ---------------------------------------------------------------------------- scale
# The span of 1.000 becomes 135 mm, the width of the SutureBot pad (135.05 mm): the dVRK ports of
# the recordings are laid out for a pad that wide, and the suture step that follows this example
# keeps them where they are. Two bounds hold at this number:
#   upper, from the thread: the mean tissue edge must not exceed sqrt(8 R r) = 3.37 mm (needle
#     radius R = 9.449 mm, thread radius r = 0.15 mm; worker_thread.py, ``eyelet_spacing``).
#     Whole body: 3.367 / 19.971 = 0.169. In the wound zone the edge is 21.40 mm per metre of
#     span: 0.157.
#   lower, from the needle and a row of stitches: the part of the belly within 37 degrees of
#     level is 0.20 span wide and has to hold the needle's circle (18.9 mm) and a wound for
#     three stitches 6 mm apart (about 20 mm): 20 / 200 = 0.10.
SCALE = 0.135

# ---------------------------------------------------------------------------- wound and stitch
# Asset units. The incision runs across the belly (along x) on the line through the apex. A cut
# along the body would close when the limbs are pulled; a cut across it opens.
WOUND_Y = -0.040
# Half the blade's length: the belly is within 37 degrees of level for |x| <= 0.10.
WOUND_HALF_LENGTH = 0.10
# The blade's reach below the apex: 31 % of the body's thickness there (0.325), more than five
# times the depth of the needle's path under the skin (1.97 mm at a 10 mm bite, in the scene).
WOUND_DEPTH = 0.10

# ---------------------------------------------------------------------------- place
# The stored frame is kept: the loader follows the file's header (up axis Z) and applies no
# rotation, which lays the body on its back, belly up, arm span along x. The wound then runs
# along x and the stitch across it along -y, like the recorded stitch on the pad (which runs
# 13.5 degrees off -y).
# The middle of the stitch, the wound's centre on the skin, is put where the middle of the
# recorded stitch is in the pad frame [m] (examples/grasp/_dvrk_layout.py,
# ``load_source_stitch``: entry (39.25, -40.28, 1.63) mm, exit (38.01, -45.54, 1.81) mm), so
# that the ports of the later suture step reach the stitch as they reach it on the pad.
PAD_STITCH_MIDPOINT = (0.038629, -0.042907, 0.001717)
# The wound's centre on the skin, asset units: x = 0, on the wound line, at the height of the apex
# (the skin is 0.00007 lower there than at the apex node, 0.009 mm in the scene).
STITCH_MIDPOINT = (0.0, WOUND_Y, BELLY_APEX[2])
# Scene position [m] of the asset's origin.
OFFSET = tuple(pad - SCALE * own for pad, own in zip(PAD_STITCH_MIDPOINT, STITCH_MIDPOINT, strict=True))
# The body's back is its thickness below the stitch: the floor of the cut scene is there, at the
# lowest point of the body as it is loaded. [m]
FLOOR_HEIGHT = OFFSET[2] - SCALE * 0.5 * EXTENT[2]

# ---------------------------------------------------------------------------- node sets
# Asset units; the rules of the limbs, the head and the belly.
# An arm: |x| beyond the thinnest section of the arm, the shoulder's neck (97 cm2 at span 1).
ARM_X = 0.20
# A hand: the outermost 0.06 of the span.
HAND_X = 0.44
# A leg: below this height the two legs are separate pieces of the mesh; above it, to about
# -0.17, they are joined.
LEG_Y = -0.25
# A foot: the lowest 0.026 of the span.
FOOT_Y = -0.29
# The head: nothing else of the body is above 0.2445 (the tops of the arms reach 0.2424).
HEAD_Y = 0.25
# The belly: |x| <= 0.12, -0.15 <= y <= 0.08, z >= 0.05.
BELLY_BOX = ((-0.12, -0.15, 0.05), (0.12, 0.08, math.inf))

# ---------------------------------------------------------------------------- lengths of the cut scene
# In the scene [m]; each from the body, or a choice with its reason.
# Reach of the contact between cut pieces: one and a half mean tissue edges. The contact asks,
# from the middle of every surface triangle, for the nearest triangle of another piece within
# this distance; a lip that has gone deeper than that into another piece is no longer pushed
# out. The factor is a choice: the ratio that the pad had (2 mm on its 1.31 mm edge) and that
# the scene this body comes from has (0.03 on 0.01997, at span 1). It is above the depth at
# which the contact's pressure reaches the tissue's modulus (30 kPa / 2e7 N/m3 = 1.5 mm): no
# contact in balance is deeper than that.
CONTACT_SEARCH_RADIUS = 1.5 * SCALE * MEAN_EDGE
# Radius of the patch of surface that one grip holds: 1.6 surface edges. A choice: the ratio
# that the pad had (2 mm on its 1.25 mm surface edge), with which the patch holds the same
# number of surface triangles, about 19 (pi r^2 over the triangle's area), among the 64 sample
# cells of a grip.
GRIP_RADIUS = 1.6 * SCALE * SURFACE_EDGE
# The grip's damper [kg/s] over its spring [N/m], a time [s]. A choice: the pad's (0.03 kg/s
# over 20 N/m), a third of a substep of 1/240 s, so that the damper's share of the step's
# matrix stays what it was beside the spring's.
GRIP_RELAXATION_TIME = 0.0015
# Stretch limits of the material law in the cut scenes; the lower one bounds the grip's speed.
MIN_STRETCH, MAX_STRETCH = 0.2, 5.0
# A node may move in one substep by an eighth of what its smallest element has left above the
# stretch limit (_src/solvers/coupled/interface/swept_tool_tissue_contact.py, "Allowed move").
MOVE_FRACTION = 0.125

# ---------------------------------------------------------------------------- start view
# The viewer's picture: 45 degrees of vertical field of view, 16 to 9, 1080 pixels high (the
# defaults of the OpenGL viewer). The screen tools size a cylinder or a sphere as 8 pixels across
# at the depth of the point that is clicked: a size on the screen, not a length of the body.
VIEW_FIELD_DEGREES, VIEW_ASPECT, VIEW_HEIGHT_PIXELS = 45.0, 16.0 / 9.0, 1080
# The body is seen from the feet and from above, so that the head is at the top of the picture
# and the arm span, with the wound line, runs across it. The elevation is a choice: halfway
# between the view from above (the wound straight on, the thickness hidden) and the view from
# the side (the thickness seen, the wound edge on).
VIEW_ELEVATION_DEGREES = 45.0
# The camera looks at a point a quarter of the span beside the body's centre, which leaves the
# left quarter of the picture to the panel (a choice of the picture, as before).
VIEW_PANEL_SHIFT = 0.25
# Distance of the camera from that point, in spans: the far hand, 0.25 + 0.5 spans from it,
# stands at 0.9 of half the picture's width.
VIEW_DISTANCE = (VIEW_PANEL_SHIFT + 0.5) / (0.9 * VIEW_ASPECT * math.tan(math.radians(0.5 * VIEW_FIELD_DEGREES)))


def grip_stiffness(young_modulus, poisson_ratio, radius=GRIP_RADIUS):
    """Stiffness of a grip's spring [N/m]: ten times that of the tissue the grip holds.

    The tissue under a held patch of radius ``a`` gives way like a half-space under a flat
    round punch, with stiffness 2 E a / (1 - nu^2). A spring ten times as stiff leaves the patch
    within a tenth of the way to the grip's target. (The pad's 20 N/m was a seventh of its own
    patch stiffness: its patch stayed behind and the spring stretched.)
    """
    return 10.0 * 2.0 * young_modulus * radius / (1.0 - poisson_ratio**2)


def grip_speed(substeps_per_second):
    """Largest speed of a grip's target [m/s]: no element can be inverted by a held node that follows it.

    The allowed move of a node per substep, ``MOVE_FRACTION`` of the smallest altitude of the
    body times what an element at rest has above the stretch limit, times the substeps of a second.
    """
    return MOVE_FRACTION * SCALE * SMALLEST_ALTITUDE * (1.0 - MIN_STRETCH) * substeps_per_second


def place(points):
    """Scene positions [m] of points given in asset units."""
    return SCALE * np.asarray(points, dtype=np.float64) + np.asarray(OFFSET)


def asset_coordinates(positions):
    """Asset units of scene positions [m]: the inverse of :func:`place`."""
    return (np.asarray(positions, dtype=np.float64) - np.asarray(OFFSET)) / SCALE


def load_body(path=ASSET):
    """Load the body as it is stored and put it into the scene.

    Returns:
        Reference positions [m] in the scene frame and the tetrahedra.
    """
    (mesh,) = load_mesh(path)
    return place(mesh.vertices), mesh.indices.copy()


def node_sets(positions):
    """Nodes of the limbs, the head and the belly, from reference positions in the scene frame [m].

    Returns:
        Index arrays by name: ``arm_plus_x``, ``arm_minus_x``, ``hand_plus_x``, ``hand_minus_x``,
        ``leg_plus_x``, ``leg_minus_x``, ``foot_plus_x``, ``foot_minus_x``, ``head``, ``belly``.
        A hand is part of its arm and a foot part of its leg.
    """
    x, y, z = asset_coordinates(positions).T
    lower, upper = BELLY_BOX
    masks = {
        "arm_plus_x": x > ARM_X,
        "arm_minus_x": x < -ARM_X,
        "hand_plus_x": x > HAND_X,
        "hand_minus_x": x < -HAND_X,
        "leg_plus_x": (y < LEG_Y) & (x > 0.0),
        "leg_minus_x": (y < LEG_Y) & (x < 0.0),
        "foot_plus_x": (y < FOOT_Y) & (x > 0.0),
        "foot_minus_x": (y < FOOT_Y) & (x < 0.0),
        "head": y > HEAD_Y,
        "belly": (x >= lower[0]) & (x <= upper[0]) & (y >= lower[1]) & (y <= upper[1]) & (z >= lower[2]),
    }
    return {name: np.flatnonzero(mask) for name, mask in masks.items()}


def wound_blade():
    """The blade of the incision: four corners of a planar quadrilateral in the scene frame [m].

    It stands on the wound line, across the belly, ``2 * WOUND_HALF_LENGTH`` long (27.0 mm). Its
    lower edge is ``WOUND_DEPTH`` below the apex (13.5 mm) and its upper edge as far above it,
    outside the body. Order: lower edge from -x to +x, then the upper edge back.
    """
    top = BELLY_APEX[2]
    low, high = top - WOUND_DEPTH, top + WOUND_DEPTH
    return place(
        (
            (-WOUND_HALF_LENGTH, WOUND_Y, low),
            (WOUND_HALF_LENGTH, WOUND_Y, low),
            (WOUND_HALF_LENGTH, WOUND_Y, high),
            (-WOUND_HALF_LENGTH, WOUND_Y, high),
        )
    )


def stitch_frame():
    """Where a stitch crosses the wound, in the scene frame.

    Returns:
        ``midpoint`` [m]: the wound's centre on the skin; ``direction``: from the entry to the
        exit, across the wound; ``wound_axis``: along the wound; ``up``: out of the skin.
    """
    return {
        "midpoint": place(STITCH_MIDPOINT),
        "direction": np.asarray((0.0, -1.0, 0.0)),
        "wound_axis": np.asarray((1.0, 0.0, 0.0)),
        "up": np.asarray((0.0, 0.0, 1.0)),
    }
