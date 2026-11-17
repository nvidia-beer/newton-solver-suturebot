# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tissue between the two jaws of a tool: which nodes the jaws carry, how hard they squeeze, how hard the tissue pulls.

The swept contact (:class:`SweptToolTissueContact`) places tissue on the capsules of a tool and
holds it there along the push direction of each node; it has no friction. This object adds
what two jaws that close on tissue do across that direction. It runs after every update of
the law, on the same accepted state, reads what the law published in that update and knows
the tools only as capsules on bodies, pairs of jaw bodies and one friction coefficient per
pair. Per pair of jaws A and B:

P1 **Catch.** For a node, the *closing direction* ``c`` is the unit vector from its foot on
   the axis of the nearest capsule of its own jaw to its foot on the axis of the nearest
   capsule of the other jaw (nearest: the smallest distance to the capsule's surface), both
   at the poses that the law predicted in this update; where the two feet coincide there is
   none. A free surface node becomes *carried* by jaw A when the law holds it on a capsule
   of A with ``push . c > 0``: it is pressed on the half of A that faces B. It stays A's
   until it leaves the pinch. A carried node that the law holds becomes *clamped*, and stays
   so until it leaves, the first time that all of these hold: it pushed back through the
   interval that ended; ``push . c >= 1 / sqrt(1 + mu^2)`` (a face that is tilted more than
   the friction angle from the closing direction pushes the tissue out instead of holding
   it); the other jaw is opposite (the foot on it is not at a free end of its capsule set);
   the faces are apart there (the distance of the two feet less the two radii is positive).
P2 **Squeeze and pull.** ``N_A = max(0, -F_A . c_A)`` with ``F_A`` the force of the law's
   wrench on body A after this update and ``c_A`` the normalised sum of the closing
   directions of A's carried nodes. ``S = min(N_A, N_B)``: the squeeze is the law's force on
   the jaws, not a force of this object. ``T = |sum over the carried nodes of both jaws of
   f_i - (f_i . c_J) c_J|`` with ``f_i`` the elastic force on node i at the accepted state
   and ``J`` its jaw: what the tissue pulls across the closing direction. The pair *clamps*
   when both jaws have a clamped node and ``S`` is above what rounding can make of no squeeze,
   ``S > u (K_c + n K_node)``: ``u`` is the float32 spacing of a tissue coordinate (at the
   largest coordinate of the tissue at rest), ``K_c`` the held-tissue stiffness of the tool
   (the law's: all its touching pieces together turn a depth into force with it), ``n`` the
   larger number of carried nodes of the two jaws and ``K_node`` the largest stiffness of a
   free surface node against a move of its own (the largest over those nodes of the sum over
   its tetrahedra of ``(lambda + 2 mu_L) V |grad N|^2``). ``S`` is a sum of forces that are
   stiffnesses times differences of stored positions, one ``K_c`` for the pieces and one
   ``K_node`` for each node that returns a reaction; a position is stored to one spacing, so
   a squeeze below that sum cannot be told from none. A clamping pair *sticks* when
   ``T <= 2 mu S`` (two faces) and *slips* otherwise.
P3 **Stick.** Tissue between two faces goes where the faces go. A carried node gets the
   velocity that takes it, in one full interval, to its anchor (its place in the frame of
   its jaw) at the pose that the law predicts for the jaw, across the closing direction
   ``c_J`` of the jaw, and along ``c_J`` by what both jaws travel alike: with ``w`` the way
   from the node to that place and ``u_J``, ``u_O`` the travel of the node's own jaw and of
   the other jaw at the node over the next interval (from the accepted to the predicted pose),

       s = minmod(u_J . c_J, u_O . c_J)       0 if the two differ in sign, else the smaller in size
       t = w - (w . c_J) c_J + s c_J          particle_qd = t / h_full

   A motion that both jaws make alike moves the clamped tissue with them whatever the
   closing direction is: for it ``u_J = u_O`` and ``t`` is the whole way, so no depth of the
   contact changes. Jaws that close or open travel in opposite senses along ``c_J`` and a
   jaw that moves while the other stands shares nothing with it: there ``s = 0`` and along
   ``c_J`` the node is the law's, with the law's bound on how far tissue can be pushed.
   The node itself is not moved here: it is held (by the law, or by this object for a node
   that the law has let go: its flag in the law's hold set and the force that balances its
   elastic force in the tissue solve), a held node keeps the velocity it is given (the hold
   leaves ``1 / HOLD_FACTOR`` of what a free node would change of it), and so the tissue
   step takes it there together with its free neighbours, and the guard of the interval
   checks that motion like any other: an interval that an element cannot follow is rejected
   and repeated at half its length, in which the node goes half of the way. Along ``c_J`` a
   node that the law holds has the place the law gave it, moved by ``s``; one that the law
   has let go (it pulls on its face) moves by ``s`` alone. The anchor becomes the place that
   the node is on its way to. The carry, the pull (P2) and the reaction (P6) are taken
   across the same direction: what the jaw pays along its travel is what the tissue receives.
P4 **Slip.** No node is carried: the jaws slide on. Carried nodes that the law holds stay
   marked and their anchors are taken again; the others leave the pinch. The largest travel
   of a carried node's anchor across ``c_J`` in the interval is added to the slip length of
   the pair.
P5 **Not clamping.** A carried node that the law holds stays marked and its anchor is taken
   again; one that the law has let go leaves the pinch. Nothing is written.
P6 **Reaction.** ``g_i`` is ``f_i`` less what the law returned of it to the tool (its part
   along the push direction that the law's reaction read, if the node pushed back). In stick
   every carried node gives its jaw ``g_i - (g_i . c_J) c_J``. The part along ``c_J`` goes to
   the node's jaw if the law holds the node (law and pinch together then return ``f_i``),
   and half to each of the two jaws if only the pinch holds it: such a node pulls on its
   own face and is between both, and with the halves the two jaws together are paid all of
   that part's work in a travel that they share (P3) and none in a closing, where they
   travel in opposite senses. In slip the part across ``c_J`` is scaled by ``2 mu S / T``:
   the jaws feel the friction limit; a node that only the pinch held has left. The force
   acts on the jaw body at the node, through the next interval, as the law's reaction
   does, in a wrench array of this object that the exchange adds beside the law's.

P7 **Letting go.** A node that leaves the pinch while the law holds it is the law's. A node
   that was never *gripped* (carried in an interval in which its pair clamped) is free as
   the law leaves it. A gripped node that the law does not hold and that carries elastic
   force is not dropped: it is *let down*. In every interval its own step is taken, that of
   one block-Jacobi iteration of the tissue step for this node alone,

       step = h^2 A_ii^-1 f        A_ii: its block of the tissue matrix without a hold,

   with ``f`` its elastic force and ``h`` the interval that the matrix holds: where its
   force would vanish with its neighbours where they are. The volume of a tetrahedron is
   linear in the place of one node, so the share of its volume that this step takes,
   ``- grad log V . step``, is exact; ``q`` is the largest share over the node's tetrahedra
   (a step that opens every element has none). While ``q`` is above ``MOVE_FRACTION`` (the
   law's share: four moving nodes of an element leave ``(7/8)^4`` of it) the node is held
   and gets the velocity of the step shortened to that share, ``(MOVE_FRACTION / q) step /
   h_full``. With ``q`` within the share the node is free: the tissue step moves it no
   further than that. It is also free when the law takes it or when it is carried again.

P8 **Reports of the carry.** Per tool, running maxima since the last reset or
   :meth:`JawTissuePinch.clear_reports`, over the carried nodes of a sticking pair that were
   carried while it clamped before: ``carry_residual``, the
   largest distance of a node from its place on its jaw at the accepted pose, across the
   closing direction of the jaw (how far tissue is from the jaws when an interval has
   ended), and ``place_travel``, the largest travel of such a place over the next interval
   (from the accepted pose of the jaw to the predicted one). Nothing in a step reads them.
   While both jaws travel alike the distance is what the prediction of the jaw's pose was
   off by and what the hold left of the move; while they close or open it also holds what
   the law moved the node on its face, across the closing direction.

What this object writes: velocities of tissue nodes of the accepted state (of carried nodes
of a sticking pair and of nodes that are let down; never a position), and two arrays of the
law, ``held`` and ``balance``, only for nodes that the law has let go and the pinch keeps or
lets down. It never writes the law's ``owner``, ``push``, ``compressive``, ``stored`` or
``wrench``. A pair that has not clamped writes nothing at all: no node is given a velocity,
kept or let down.
Storage is fixed, every launch is issued in every update, and nothing reads the device but
:meth:`diagnostics`.
"""

import numpy as np
import warp as wp

from .swept_tool_tissue_contact import MOVE_FRACTION

# States of a pair of jaws after an update.
NOT_CLAMPING, STICK, SLIP = 0, 1, 2
STATE_NAMES = ("not clamping", "stick", "slip")


@wp.struct
class _Foot:
    """The point of a body's capsule set that is nearest to a node."""

    point: wp.vec3
    radius: float
    free_end: int
    found: int


@wp.func
def _nearest_foot(
    x: wp.vec3,
    body: int,
    pose: wp.transform,
    capsule_body: wp.array[int],
    capsule_start: wp.array[wp.vec3],
    capsule_end: wp.array[wp.vec3],
    capsule_radius: wp.array[float],
    capsule_free: wp.array[wp.vec2i],
):
    """The foot of ``x`` on the axis of the capsule of ``body`` whose surface is nearest to it.

    The capsules are walked in their order and the first nearest one is kept. ``free_end`` is
    set where the foot is an end of the axis that lies in no other capsule of the body.
    """
    foot = _Foot()
    foot.found = 0
    foot.free_end = 0
    foot.radius = 0.0
    foot.point = wp.vec3(0.0)
    best = float(wp.inf)
    for capsule in range(capsule_body.shape[0]):
        if capsule_body[capsule] != body:
            continue
        start = wp.transform_point(pose, capsule_start[capsule])
        axis = wp.transform_point(pose, capsule_end[capsule]) - start
        length_sq = wp.length_sq(axis)
        place = float(0.0)
        if length_sq > 0.0:
            place = wp.clamp(wp.dot(x - start, axis) / length_sq, 0.0, 1.0)
        point = start + place * axis
        gap = wp.length(x - point) - capsule_radius[capsule]
        if gap < best:
            best = gap
            foot.found = 1
            foot.point = point
            foot.radius = capsule_radius[capsule]
            free = capsule_free[capsule]
            foot.free_end = 0
            if place <= 0.0 and free[0] != 0:
                foot.free_end = 1
            if place >= 1.0 and free[1] != 0:
                foot.free_end = 1
    return foot


@wp.func
def _shared(first: float, second: float):
    """What two travels along one direction share: zero if they differ in sign, else the one of smaller size."""
    if first * second <= 0.0:
        return 0.0
    if wp.abs(first) < wp.abs(second):
        return first
    return second


@wp.kernel(enable_backward=False)
def _begin(
    body_closing_sum: wp.array[wp.vec3],
    body_carried: wp.array[int],
    body_clamped: wp.array[int],
    wrench: wp.array[wp.spatial_vector],
    tool_pull: wp.array[wp.vec3],
    tool_travel: wp.array[float],
):
    """Clear the sums of one update and the reaction of the last one, per body and per tool."""
    index = wp.tid()
    if index < body_closing_sum.shape[0]:
        body_closing_sum[index] = wp.vec3(0.0)
        body_carried[index] = 0
        body_clamped[index] = 0
        wrench[index] = wp.spatial_vector()
    if index < tool_pull.shape[0]:
        tool_pull[index] = wp.vec3(0.0)
        tool_travel[index] = 0.0


@wp.kernel(enable_backward=False)
def _catch(
    nodes: wp.array[int],
    particle_q: wp.array[wp.vec3],
    held: wp.array[int],
    owner: wp.array[int],
    push: wp.array[wp.vec3],
    compressive: wp.array[int],
    predicted: wp.array[wp.transform],
    body_tool: wp.array[int],
    body_other: wp.array[int],
    friction: wp.array[float],
    capsule_body: wp.array[int],
    capsule_start: wp.array[wp.vec3],
    capsule_end: wp.array[wp.vec3],
    capsule_radius: wp.array[float],
    capsule_free: wp.array[wp.vec2i],
    jaw: wp.array[int],
    clamped: wp.array[int],
    gripped: wp.array[int],
    anchor: wp.array[wp.vec3],
    walk: wp.array[int],
    closing: wp.array[wp.vec3],
    body_closing_sum: wp.array[wp.vec3],
    body_carried: wp.array[int],
    body_clamped: wp.array[int],
):
    """P1 per free surface node: its closing direction, whether it is caught and whether it is clamped.

    ``held``, ``owner``, ``push`` and ``compressive`` are the law's, as its update of this
    accepted state left them. A carried node without a closing direction keeps a zero one:
    it leaves the pinch in the kernel that follows.
    """
    particle = nodes[wp.tid()]
    closing[particle] = wp.vec3(0.0)
    body = jaw[particle]
    fresh = int(0)
    if body < 0:
        if held[particle] == 0:
            return
        body = owner[particle]
        if body < 0:
            return
        if body_tool[body] < 0:
            return
        fresh = 1
    other = body_other[body]
    x = particle_q[particle]
    own = _nearest_foot(
        x, body, predicted[body], capsule_body, capsule_start, capsule_end, capsule_radius, capsule_free
    )
    far = _nearest_foot(
        x, other, predicted[other], capsule_body, capsule_start, capsule_end, capsule_radius, capsule_free
    )
    offset = far.point - own.point
    distance = wp.length(offset)
    if own.found == 0 or far.found == 0 or distance <= 0.0:
        return
    c = offset / distance
    cosine = wp.dot(push[particle], c)
    if fresh == 1:
        if cosine <= 0.0:
            return
        jaw[particle] = body
        clamped[particle] = 0
        gripped[particle] = 0
        walk[particle] = 0
        anchor[particle] = wp.transform_point(wp.transform_inverse(predicted[body]), x)
    closing[particle] = c
    if clamped[particle] == 0 and held[particle] != 0 and compressive[particle] != 0:
        mu = friction[body_tool[body]]
        if cosine >= 1.0 / wp.sqrt(1.0 + mu * mu) and far.free_end == 0 and distance - own.radius - far.radius > 0.0:
            clamped[particle] = 1
    wp.atomic_add(body_closing_sum, body, c)
    wp.atomic_add(body_carried, body, 1)
    if clamped[particle] != 0:
        wp.atomic_add(body_clamped, body, 1)


@wp.kernel(enable_backward=False)
def _squeeze(
    body_tool: wp.array[int],
    body_closing_sum: wp.array[wp.vec3],
    law_wrench: wp.array[wp.spatial_vector],
    body_closing: wp.array[wp.vec3],
    body_normal: wp.array[float],
):
    """P2 per jaw body: its closing direction ``c_J`` and what the law presses on it along that direction [N]."""
    body = wp.tid()
    body_closing[body] = wp.vec3(0.0)
    body_normal[body] = 0.0
    if body_tool[body] < 0:
        return
    total = body_closing_sum[body]
    length = wp.length(total)
    if length > 0.0:
        c = total / length
        body_closing[body] = c
        body_normal[body] = wp.max(0.0, -wp.dot(wp.spatial_top(law_wrench[body]), c))


@wp.kernel(enable_backward=False)
def _pull(
    nodes: wp.array[int],
    elastic: wp.array[wp.vec3],
    jaw: wp.array[int],
    closing: wp.array[wp.vec3],
    body_tool: wp.array[int],
    body_closing: wp.array[wp.vec3],
    tool_pull: wp.array[wp.vec3],
):
    """P2 per carried node: its elastic force across the closing direction of its jaw, summed per tool [N]."""
    particle = nodes[wp.tid()]
    body = jaw[particle]
    if body < 0:
        return
    if wp.length_sq(closing[particle]) == 0.0:
        return
    c = body_closing[body]
    force = elastic[particle]
    wp.atomic_add(tool_pull, body_tool[body], force - wp.dot(force, c) * c)


@wp.kernel(enable_backward=False)
def _decide(
    tool_jaws: wp.array[wp.vec2i],
    friction: wp.array[float],
    squeeze_floor: wp.array[float],
    squeeze_floor_per_node: wp.array[float],
    body_carried: wp.array[int],
    body_clamped: wp.array[int],
    body_normal: wp.array[float],
    tool_pull: wp.array[wp.vec3],
    squeeze: wp.array[float],
    pull: wp.array[float],
    state: wp.array[int],
    scale: wp.array[float],
    counters: wp.array2d[int],
):
    """P2 per tool: squeeze, pull, and whether the pair does not clamp, sticks or slips; counts the intervals."""
    tool = wp.tid()
    pair = tool_jaws[tool]
    held = float(0.0)
    pulled = wp.length(tool_pull[tool])
    decision = int(0)
    share = float(0.0)
    if pair[0] >= 0:
        if body_clamped[pair[0]] > 0 and body_clamped[pair[1]] > 0:
            held = wp.min(body_normal[pair[0]], body_normal[pair[1]])
        # A squeeze that rounding of the stored positions can make of none does not clamp: u (K_c + n K_node).
        carried = wp.max(body_carried[pair[0]], body_carried[pair[1]])
        if held > squeeze_floor[tool] + float(carried) * squeeze_floor_per_node[tool]:
            limit = 2.0 * friction[tool] * held
            if pulled <= limit:
                decision = 1
                share = 1.0
            else:
                decision = 2
                share = limit / pulled
            counters[tool, 0] = counters[tool, 0] + 1
            counters[tool, decision] = counters[tool, decision] + 1
    squeeze[tool] = held
    pull[tool] = pulled
    state[tool] = decision
    scale[tool] = share


@wp.kernel(enable_backward=False)
def _carry(
    nodes: wp.array[int],
    elastic: wp.array[wp.vec3],
    compressive: wp.array[int],
    law_push: wp.array[wp.vec3],
    predicted: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_tool: wp.array[int],
    body_other: wp.array[int],
    body_closing: wp.array[wp.vec3],
    pair_state: wp.array[int],
    scale: wp.array[float],
    closing: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    full_step: float,
    step_seconds: wp.array[wp.float64],
    pristine: wp.array[wp.mat33f],
    tet_indices: wp.array2d[int],
    tet_materials: wp.array2d[float],
    node_tet_offset: wp.array[int],
    node_tets: wp.array[int],
    particle_qd: wp.array[wp.vec3],
    held: wp.array[int],
    balance: wp.array[wp.vec3],
    jaw: wp.array[int],
    clamped: wp.array[int],
    gripped: wp.array[int],
    anchor: wp.array[wp.vec3],
    push_read: wp.array[wp.vec3],
    walk: wp.array[int],
    wrench: wp.array[wp.spatial_vector],
    tool_travel: wp.array[float],
    carry_residual: wp.array[float],
    place_travel: wp.array[float],
):
    """P3 to P8 per free surface node: stick, slip or not clamping, by the decision of its pair; a released node is let down.

    ``held`` is the law's flag as its update of this accepted state left it; this kernel sets
    it, with the balance force, for a carried node that the law has let go and that sticks.
    ``push_read`` holds the push direction that the law's reaction of this update read (the
    law's push of the update before); it takes the law's push of this update for the next one.
    ``particle_q`` is read only (the node, and for a node that is let down the nodes of its
    tetrahedra): no thread of this kernel writes a position. ``pristine`` holds, in the order
    of ``nodes``, the diagonal block of each node in the tissue matrix without a hold, and
    ``step_seconds`` the interval that this matrix holds; ``node_tet_offset`` and
    ``node_tets`` list the tetrahedra at each node (four times the tetrahedron plus the
    corner of the node in it).
    """
    index = wp.tid()
    particle = nodes[index]
    read = push_read[particle]
    push_read[particle] = law_push[particle]
    body = jaw[particle]
    by_law = held[particle] != 0
    force = elastic[particle]
    x = particle_q[particle]
    if body >= 0:
        c = closing[particle]
        tool = body_tool[body]
        decision = pair_state[tool]
        leaves = int(0)
        if decision == 0 or wp.length_sq(c) == 0.0:
            # P5: not clamping (or no closing direction any more). Nothing is written for a node that stays.
            if by_law and wp.length_sq(c) > 0.0:
                anchor[particle] = wp.transform_point(wp.transform_inverse(predicted[body]), x)
            else:
                leaves = 1
        else:
            # The pair clamps: this node was carried while it did.
            was_gripped = gripped[particle]
            gripped[particle] = 1
            # P6: what the law did not return of the node's elastic force, along and across the closing
            # direction of the jaw.
            rest = force
            if compressive[particle] != 0:
                rest = force - wp.dot(force, read) * read
            jaw_closing = body_closing[body]
            along = wp.dot(rest, jaw_closing) * jaw_closing
            across = rest - along
            # The way to the node's place on its jaw at the predicted pose, across the closing direction of the jaw.
            place = wp.transform_point(predicted[body], anchor[particle])
            travel = place - x
            travel = travel - wp.dot(travel, jaw_closing) * jaw_closing
            applied = wp.vec3(0.0)
            if decision == 1:
                if was_gripped != 0:
                    # A report (P8): how far the node is from its place on the jaw now that the interval has ended,
                    # across the closing direction of the jaw, and how far that place travels in the next one.
                    behind = wp.transform_point(body_q[body], anchor[particle]) - x
                    wp.atomic_max(carry_residual, tool, wp.length(behind - wp.dot(behind, jaw_closing) * jaw_closing))
                    wp.atomic_max(place_travel, tool, wp.length(place - x - behind))
                # P3: along the closing direction the node makes the travel that both jaws share at it over the
                # next interval (none where they close, open, or only one of them moves).
                other = body_other[body]
                own_travel = (
                    wp.transform_point(predicted[body], wp.transform_point(wp.transform_inverse(body_q[body]), x)) - x
                )
                other_travel = (
                    wp.transform_point(predicted[other], wp.transform_point(wp.transform_inverse(body_q[other]), x)) - x
                )
                travel = (
                    travel + _shared(wp.dot(own_travel, jaw_closing), wp.dot(other_travel, jaw_closing)) * jaw_closing
                )
                # The velocity that takes the held node that way in one full interval; the tissue step moves it,
                # with its neighbours, and the guard of the interval checks that motion.
                particle_qd[particle] = travel / full_step
                if not by_law:
                    held[particle] = 1
                    balance[particle] = -force
                # The anchor is the place that the node is on its way to.
                anchor[particle] = wp.transform_point(wp.transform_inverse(predicted[body]), x + travel)
                applied = across
                if by_law:
                    applied = applied + along
                else:
                    # P6: a node that only the pinch holds gives its part along the closing direction half to
                    # each jaw.
                    half = 0.5 * along
                    applied = applied + half
                    lever = x - wp.transform_point(body_q[other], body_com[other])
                    wp.atomic_add(wrench, other, wp.spatial_vector(half, wp.cross(lever, half)))
            else:
                # P4: the jaws slide on; the tool feels the friction limit.
                wp.atomic_max(tool_travel, tool, wp.length(travel))
                applied = scale[tool] * across
                if by_law:
                    applied = applied + along
                    anchor[particle] = wp.transform_point(wp.transform_inverse(predicted[body]), x)
                else:
                    leaves = 1
            lever = x - wp.transform_point(body_q[body], body_com[body])
            wp.atomic_add(wrench, body, wp.spatial_vector(applied, wp.cross(lever, applied)))
        if leaves == 1:
            # P7: a node that leaves while the law does not hold it. If its pair clamped while it was carried and
            # it carries a force, it is not dropped: it is let down, from this update on.
            if gripped[particle] != 0 and not by_law and wp.length_sq(force) > 0.0:
                walk[particle] = 1
            jaw[particle] = -1
            clamped[particle] = 0
            gripped[particle] = 0
    if jaw[particle] < 0 and walk[particle] != 0:
        # P7: the node's own step of one block-Jacobi iteration of the tissue step, and the largest share of
        # the volume of one of its tetrahedra that this step takes (exact: the volume is linear in one node).
        seconds = float(step_seconds[0])
        step = (seconds * seconds) * (wp.inverse(pristine[index]) * force)
        share = float(0.0)
        for entry in range(node_tet_offset[index], node_tet_offset[index + 1]):
            code = node_tets[entry]
            element = code // 4
            corner = code - 4 * element
            # A tetrahedron without material carries nothing (the activation contract of the tissue step).
            if (
                tet_materials[element, 0] == 0.0
                and tet_materials[element, 1] == 0.0
                and tet_materials[element, 2] == 0.0
            ):
                continue
            origin = particle_q[tet_indices[element, 0]]
            e1 = particle_q[tet_indices[element, 1]] - origin
            e2 = particle_q[tet_indices[element, 2]] - origin
            e3 = particle_q[tet_indices[element, 3]] - origin
            g1 = wp.cross(e2, e3)
            g2 = wp.cross(e3, e1)
            g3 = wp.cross(e1, e2)
            # Six times the volume and its gradient at the node.
            volume = wp.dot(e1, g1)
            gradient = -(g1 + g2 + g3)
            if corner == 1:
                gradient = g1
            elif corner == 2:
                gradient = g2
            elif corner == 3:
                gradient = g3
            share = wp.max(share, -wp.dot(gradient, step) / volume)
        if by_law or share <= MOVE_FRACTION:
            # Free: the law holds it, or its own step takes no element more than the share.
            walk[particle] = 0
        else:
            # Held, and let down by its step shortened to the share.
            particle_qd[particle] = step * (MOVE_FRACTION / share) / full_step
            held[particle] = 1
            balance[particle] = -force


@wp.kernel(enable_backward=False)
def _finish(tool_travel: wp.array[float], slip_length: wp.array[float]):
    """P4 per tool: the travel of the jaws over tissue that slipped in this interval joins the slip length [m]."""
    tool = wp.tid()
    slip_length[tool] = slip_length[tool] + tool_travel[tool]


@wp.kernel(enable_backward=False)
def _add_reaction(wrench: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector]):
    """Add the pinch's reaction to the body forces, only where there is one: a body without keeps every bit."""
    body = wp.tid()
    value = wrench[body]
    if wp.length_sq(wp.spatial_top(value)) > 0.0 or wp.length_sq(wp.spatial_bottom(value)) > 0.0:
        body_f[body] = body_f[body] + value


def largest_node_stiffness(model, nodes):
    """The largest stiffness of one of ``nodes`` against a move of its own [N/m], an upper bound from the rest state.

    For a node ``i`` the sum over its tetrahedra of ``(lambda + 2 mu_L) V |grad N_i|^2``: the
    diagonal block of linear elasticity at rest is the sum of ``V (mu_L |g|^2 I + (lambda +
    mu_L) g g^T)`` with ``g = grad N_i``, and its largest eigenvalue is at most that sum.
    ``model.tet_poses`` holds the inverse rest edge matrix of each tetrahedron (its rows are
    the gradients of the shape functions of nodes 1 to 3, minus their sum that of node 0),
    ``model.tet_materials`` the Lame parameters ``mu_L`` and ``lambda``. A tetrahedron without
    material carries none (the activation contract of the tissue step) and is left out.

    Args:
        model: Model of the tissue tetrahedra.
        nodes: Particles to take the largest over.
    """
    indices = model.tet_indices.numpy().reshape(-1, 4)
    materials = model.tet_materials.numpy().astype(float)
    active = np.flatnonzero(np.any(materials[:, :3] != 0.0, axis=1))
    inverse = model.tet_poses.numpy().astype(float)[active]
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    volume = 1.0 / (6.0 * np.abs(np.linalg.det(inverse)))
    modulus = materials[active, 1] + 2.0 * materials[active, 0]
    stiffness = np.zeros(model.particle_count)
    np.add.at(stiffness, indices[active], (modulus * volume)[:, None] * np.einsum("tij,tij->ti", gradients, gradients))
    return float(stiffness[np.asarray(nodes, dtype=np.int64)].max())


def free_capsule_ends(bodies, endpoints, radii):
    """For each capsule, whether each end of its axis is a free end of the capsule set of its body.

    An end that lies inside another capsule of the same body (nearer to that capsule's axis
    than its radius) is where two cells of the set meet, not an end of the part.

    Args:
        bodies: Body of each capsule, shape (capsules,).
        endpoints: End points of each capsule axis in the body frame [m], shape (capsules, 2, 3).
        radii: Capsule radii [m].

    Returns:
        Flags, shape (capsules, 2): nonzero for a free end.
    """
    bodies = np.asarray(bodies)
    endpoints = np.asarray(endpoints, dtype=float).reshape(-1, 2, 3)
    radii = np.asarray(radii, dtype=float)
    free = np.ones((len(bodies), 2), dtype=np.int32)
    for capsule in range(len(bodies)):
        for other in np.flatnonzero(bodies == bodies[capsule]):
            if other == capsule:
                continue
            start, end = endpoints[other]
            axis = end - start
            length_sq = float(axis @ axis)
            for side in range(2):
                point = endpoints[capsule, side]
                place = float(np.clip((point - start) @ axis / length_sq, 0.0, 1.0)) if length_sq > 0.0 else 0.0
                if np.linalg.norm(point - start - place * axis) < radii[other]:
                    free[capsule, side] = 0
    return free


class JawTissuePinch:
    """Carried and clamped tissue nodes between the jaws of tools, on top of a swept tool/tissue contact.

    Args:
        model: Model of the bodies and of the tissue particles.
        law: The :class:`SweptToolTissueContact` whose update this object follows.
        capsule_bodies: Body of each tool capsule, shape (capsules,).
        capsule_endpoints: End points of each capsule axis in the body frame [m], shape (capsules, 2, 3).
        capsule_radii: Capsule radii [m].
        jaw_bodies: For each tool, the bodies of its two jaws, or an empty tuple for a tool without a pinch.
        friction: For each tool, the friction coefficient between its jaws and the tissue
            (ignored for a tool without a pinch).
        held_stiffness: For each tool, the held-tissue stiffness ``K_c`` of the law [N/m]
            (effective mass over the full interval squared); :meth:`set_stiffness` replaces it.

    Attributes:
        jaw: Per particle, the jaw body that carries the node, or -1.
        clamped: Per particle, nonzero for a carried node that was found clamped.
        friction: Per tool, the friction coefficient: device data, see :meth:`set_friction`.
        squeeze_floor: Per tool, ``u K_c`` [N]: device data, see :meth:`set_stiffness`.
        squeeze_floor_per_node: Per tool, ``u K_node`` [N]: what each carried node adds to the
            smallest squeeze that clamps.
        counters: Per tool, the number of intervals in which its pair clamped, stuck and slipped.
        wrench: Reaction of the carried tissue on every body [N, N m], force first, torque about
            the centre of mass, world frame; :meth:`add_forces` adds it to the body forces.
        slip_length: Per tool, how far its jaws have travelled over clamped tissue that slipped [m].
        carry_residual: Per tool, the largest distance of a carried node of the sticking pair from its
            place on its jaw at an accepted state, across the closing direction of the jaw [m] (P8).
        place_travel: Per tool, the largest travel of such a place over one interval [m] (P8).
    """

    def __init__(
        self, model, law, capsule_bodies, capsule_endpoints, capsule_radii, jaw_bodies, friction, held_stiffness
    ):
        device = model.device
        self.model, self.law = model, law
        bodies = np.asarray(capsule_bodies, dtype=np.int64)
        endpoints = np.asarray(capsule_endpoints, dtype=float).reshape(-1, 2, 3)
        radii = np.asarray(capsule_radii, dtype=float)
        if len(bodies) != len(endpoints) or len(bodies) != len(radii):
            raise ValueError("One body, two end points and one radius per capsule")
        self.tool_count = len(jaw_bodies)
        friction = np.asarray(friction, dtype=float)
        if friction.shape != (self.tool_count,):
            raise ValueError("One friction coefficient per tool")
        body_tool = np.full(model.body_count, -1, dtype=np.int32)
        body_other = np.full(model.body_count, -1, dtype=np.int32)
        pairs = np.full((self.tool_count, 2), -1, dtype=np.int32)
        for tool, pair in enumerate(jaw_bodies):
            if not len(pair):
                continue
            if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= model.body_count:
                raise ValueError("A pinch needs two different jaw bodies of the model per tool")
            if not np.isfinite(friction[tool]) or friction[tool] < 0.0:
                raise ValueError("The friction coefficient of a pair of jaws must be finite and nonnegative")
            for body, other in (pair, pair[::-1]):
                if body_tool[body] >= 0:
                    raise ValueError("A body is the jaw of one tool only")
                if not np.any(bodies == body):
                    raise ValueError("Every jaw body needs a capsule")
                body_tool[body], body_other[body] = tool, other
            pairs[tool] = pair
        self.jaw_bodies_host = pairs
        # Only the capsules on jaw bodies are walked.
        on_jaw = body_tool[bodies] >= 0
        free = free_capsule_ends(bodies[on_jaw], endpoints[on_jaw], radii[on_jaw])

        def array(values, dtype):
            return wp.array(values, dtype=dtype, device=device)

        self._capsule_body = array(bodies[on_jaw].astype(np.int32), int)
        self._capsule_start = array(endpoints[on_jaw, 0], wp.vec3)
        self._capsule_end = array(endpoints[on_jaw, 1], wp.vec3)
        self._capsule_radius = array(radii[on_jaw], float)
        self._capsule_free = array(free, wp.vec2i)
        self.capsule_count = int(on_jaw.sum())
        self._body_tool, self._body_other = array(body_tool, int), array(body_other, int)
        self._tool_jaws = array(pairs, wp.vec2i)
        # Device data: a test changes it by a host write at a reset (set_friction), as the law's inverse mass.
        self.friction = array(np.where(pairs[:, 0] >= 0, friction, 0.0), float)
        # The smallest squeeze that clamps, u (K_c + n K_node): the spacing of a stored tissue coordinate (float32,
        # at the largest coordinate of the tissue at rest) and the largest stiffness of a free surface node are
        # host values of construction; K_c is the tool's and follows set_stiffness.
        self.position_spacing = float(np.spacing(np.float32(np.abs(model.particle_q.numpy()).max())))
        self.node_stiffness = largest_node_stiffness(model, law.nodes_host)
        # The tetrahedra at each free surface node, in the order of the law's nodes (P7): four times the
        # tetrahedron plus the corner of the node in it.
        corners = np.full(model.particle_count, -1, dtype=np.int64)
        corners[law.nodes_host] = np.arange(law.node_count)
        corners = corners[model.tet_indices.numpy().reshape(-1, 4)]
        owned = np.argwhere(corners >= 0)
        owner = corners[owned[:, 0], owned[:, 1]]
        owned = owned[np.lexsort((owned[:, 1], owned[:, 0], owner))]
        offsets = np.concatenate(([0], np.cumsum(np.bincount(owner, minlength=law.node_count))))
        self._node_tet_offset = array(offsets.astype(np.int32), int)
        self._node_tets = array((4 * owned[:, 0] + owned[:, 1]).astype(np.int32), int)
        self.squeeze_floor = wp.zeros(self.tool_count, dtype=float, device=device)
        self.squeeze_floor_per_node = array(
            np.full(self.tool_count, self.position_spacing * self.node_stiffness), float
        )
        self.set_stiffness(held_stiffness)

        def zeros(count, dtype):
            return wp.zeros(count, dtype=dtype, device=device)

        particles, body_count, tools = model.particle_count, model.body_count, self.tool_count
        # State between intervals.
        self.jaw = wp.full(particles, -1, dtype=int, device=device)
        self.clamped = zeros(particles, int)
        self.gripped = zeros(particles, int)
        self.anchor = zeros(particles, wp.vec3)
        self.walk = zeros(particles, int)
        self.squeeze, self.pull = zeros(tools, float), zeros(tools, float)
        self.pair_state, self.scale = zeros(tools, int), zeros(tools, float)
        self.counters = wp.zeros((tools, 3), dtype=int, device=device)
        self.push_read = zeros(particles, wp.vec3)
        self.wrench = zeros(body_count, wp.spatial_vector)
        self.slip_length = zeros(tools, float)
        self.carry_residual, self.place_travel = zeros(tools, float), zeros(tools, float)
        # Of the last update, kept for a report: per body its closing direction, the law's force along it and
        # its carried and clamped nodes as the decision counted them.
        self.body_closing, self.body_normal = zeros(body_count, wp.vec3), zeros(body_count, float)
        self.body_carried, self.body_clamped = zeros(body_count, int), zeros(body_count, int)
        # Scratch of one update.
        self._closing = zeros(particles, wp.vec3)
        self._body_closing_sum = zeros(body_count, wp.vec3)
        self._tool_pull = zeros(tools, wp.vec3)
        self._tool_travel = zeros(tools, float)

    @property
    def state(self) -> tuple:
        """The arrays that carry over from one accepted interval to the next, and those a report reads."""
        return (
            self.jaw,
            self.clamped,
            self.gripped,
            self.anchor,
            self.walk,
            self.squeeze,
            self.pull,
            self.pair_state,
            self.scale,
            self.counters,
            self.push_read,
            self.wrench,
            self.slip_length,
            self.body_closing,
            self.body_normal,
            self.body_carried,
            self.body_clamped,
            self.carry_residual,
            self.place_travel,
        )

    def set_friction(self, friction):
        """Replace the friction coefficient of each tool (a host write of device data, outside the step)."""
        friction = np.asarray(friction, dtype=float)
        if friction.ndim == 0:
            friction = np.full(self.tool_count, float(friction))
        if friction.shape != (self.tool_count,) or not np.isfinite(friction).all() or friction.min() < 0.0:
            raise ValueError("The friction coefficients must be finite and nonnegative, one per tool")
        self.friction.assign(np.where(self.jaw_bodies_host[:, 0] >= 0, friction, 0.0).astype(np.float32))

    def set_stiffness(self, held_stiffness):
        """Replace the held-tissue stiffness ``K_c`` of each tool [N/m] (a host write of device data, outside the step).

        The exchange calls it with the effective masses of the tools over the full interval
        squared, at construction and after every placement of the tools.
        """
        stiffness = np.asarray(held_stiffness, dtype=float)
        if stiffness.shape != (self.tool_count,) or not np.isfinite(stiffness).all() or stiffness.min() <= 0.0:
            raise ValueError("The held-tissue stiffnesses must be finite and positive, one per tool")
        self.held_stiffness_host = stiffness.copy()
        self.squeeze_floor.assign((self.position_spacing * stiffness).astype(np.float32))

    def clear_reports(self):
        """Start the two reports of the carry (P8) anew: a host call between two steps, for a measurement over a span."""
        self.carry_residual.zero_()
        self.place_travel.zero_()

    def update(self, body_q, particle_q, particle_qd):
        """After the law's update at an accepted state: catch, squeeze and pull, decision, stick or slip, reaction.

        Reads the law's ``held``, ``owner``, ``push``, ``compressive`` and ``wrench`` as that
        update left them, its scratch of the same update (the elastic forces of the accepted
        state, ``_elastic``, and the predicted poses of the bodies, ``_predicted``), its full
        interval, the interval of the accepted step and the blocks of the tissue matrix
        without a hold (``holds.pristine``). Writes ``particle_qd`` of carried nodes of a pair
        that sticks and of nodes that are let down, the law's ``held`` and ``balance`` for
        those of them that the law has let go, and :attr:`wrench`; ``particle_q`` is only
        read. The caller applies the law's holds afterwards.
        """
        model, law, device = self.model, self.law, self.model.device
        wp.launch(
            _begin,
            max(model.body_count, self.tool_count),
            outputs=[
                self._body_closing_sum,
                self.body_carried,
                self.body_clamped,
                self.wrench,
                self._tool_pull,
                self._tool_travel,
            ],
            device=device,
        )
        wp.launch(
            _catch,
            law.node_count,
            inputs=[
                law._nodes,
                particle_q,
                law.held,
                law.owner,
                law.push,
                law.compressive,
                law._predicted,
                self._body_tool,
                self._body_other,
                self.friction,
                self._capsule_body,
                self._capsule_start,
                self._capsule_end,
                self._capsule_radius,
                self._capsule_free,
            ],
            outputs=[
                self.jaw,
                self.clamped,
                self.gripped,
                self.anchor,
                self.walk,
                self._closing,
                self._body_closing_sum,
                self.body_carried,
                self.body_clamped,
            ],
            device=device,
        )
        wp.launch(
            _squeeze,
            model.body_count,
            inputs=[self._body_tool, self._body_closing_sum, law.wrench],
            outputs=[self.body_closing, self.body_normal],
            device=device,
        )
        wp.launch(
            _pull,
            law.node_count,
            inputs=[law._nodes, law._elastic, self.jaw, self._closing, self._body_tool, self.body_closing],
            outputs=[self._tool_pull],
            device=device,
        )
        wp.launch(
            _decide,
            self.tool_count,
            inputs=[
                self._tool_jaws,
                self.friction,
                self.squeeze_floor,
                self.squeeze_floor_per_node,
                self.body_carried,
                self.body_clamped,
                self.body_normal,
                self._tool_pull,
            ],
            outputs=[self.squeeze, self.pull, self.pair_state, self.scale, self.counters],
            device=device,
        )
        wp.launch(
            _carry,
            law.node_count,
            inputs=[
                law._nodes,
                law._elastic,
                law.compressive,
                law.push,
                law._predicted,
                body_q,
                model.body_com,
                self._body_tool,
                self._body_other,
                self.body_closing,
                self.pair_state,
                self.scale,
                self._closing,
                particle_q,
                law.full_step,
                law._step_seconds,
                law.holds.pristine,
                model.tet_indices,
                model.tet_materials,
                self._node_tet_offset,
                self._node_tets,
            ],
            outputs=[
                particle_qd,
                law.held,
                law.balance,
                self.jaw,
                self.clamped,
                self.gripped,
                self.anchor,
                self.push_read,
                self.walk,
                self.wrench,
                self._tool_travel,
                self.carry_residual,
                self.place_travel,
            ],
            device=device,
        )
        wp.launch(_finish, self.tool_count, inputs=[self._tool_travel], outputs=[self.slip_length], device=device)

    def add_forces(self, body_f):
        """Add the reaction of the carried tissue to the body forces; does not change the state of the pinch."""
        wp.launch(_add_reaction, self.model.body_count, inputs=[self.wrench, body_f], device=self.model.device)

    def reset(self):
        """No node carried, nothing counted."""
        for array in self.state:
            array.zero_()
        self.jaw.fill_(-1)

    def modules(self):
        """Kernel modules that this object launches, with their block sizes, for a start-up that loads by name."""
        return ((__name__, (256,)),)

    def parameters(self):
        """Host values of the pinch, without a device read."""
        return {
            "rule": "tissue that the swept contact holds on the inner half of a jaw is carried by that jaw; the "
            "pair clamps when both jaws have a carried node inside the friction cone about the closing direction "
            "and the contact presses on both with more than rounding can make of no squeeze; it sticks while the "
            "pull across the closing direction is at most 2 mu S, with S the smaller of the contact's forces on "
            "the two jaws along the closing direction",
            "jaw_bodies": self.jaw_bodies_host.tolist(),
            "jaw_capsules": self.capsule_count,
            "clamp_rule": "S > u (K_c + n K_node): u the float32 spacing of a tissue coordinate, K_c the held-tissue "
            "stiffness of the tool, n the larger number of carried nodes of the two jaws, K_node the largest "
            "stiffness of a free surface node",
            "position_spacing_m": self.position_spacing,
            "node_stiffness_N_m": self.node_stiffness,
            "held_stiffness_N_m": self.held_stiffness_host.tolist(),
            "squeeze_floor_N": (self.position_spacing * self.held_stiffness_host).tolist(),
            "squeeze_floor_per_node_N": self.position_spacing * self.node_stiffness,
        }

    def diagnostics(self):
        """Explicit host readback for reports and tests; never called by the step.

        Per tool: carried and clamped nodes of each jaw as they are marked now, the law's
        force on each jaw along its closing direction, squeeze, pull, the limit ``2 mu S``, the
        state of the pair, the counted intervals, the slip length and the two reports of the
        carry (``carry_residual_m``, ``place_travel_m``). ``carried_particles``: per
        tool and jaw the marked particles. ``kept_nodes``: carried nodes that only the pinch
        holds (the law has let them go); ``walking_nodes``: released nodes that are let down
        (P7). ``reaction_N``: per tool, the force of the pinch on each jaw.
        For a measurement from the tissue side, per tool and jaw and in the order of
        ``carried_particles``: ``carried_elastic_N`` (the elastic force on each carried node at
        the accepted state), ``carried_by_law`` (whether the law holds it) and
        ``carried_law_part_N`` (the part of that force along the law's push direction, for a
        node that pushed back: what the law's reaction returns to the tool).
        """
        jaw, clamped = self.jaw.numpy(), self.clamped.numpy()
        friction = self.friction.numpy().astype(float)
        squeeze, pull = self.squeeze.numpy().astype(float), self.pull.numpy().astype(float)
        state, counters = self.pair_state.numpy(), self.counters.numpy()
        normal, closing = self.body_normal.numpy().astype(float), self.body_closing.numpy().astype(float)
        pairs = self.jaw_bodies_host
        carried = [[np.flatnonzero(jaw == body) for body in pair] if pair[0] >= 0 else [] for pair in pairs]
        wrench = self.wrench.numpy().astype(float)
        # A node that the law holds has the body of its capsule as owner; one that only the pinch holds has none:
        # it is kept (carried) or walked.
        owner = self.law.owner.numpy()
        walking = (self.walk.numpy() != 0) & (jaw < 0)
        kept = (self.law.held.numpy() != 0) & (owner < 0) & ~walking
        elastic = self.law._elastic.numpy().astype(float)
        push = self.law.push.numpy().astype(float)
        pushing = self.law.compressive.numpy() != 0
        returned = np.where(pushing[:, None], np.einsum("ij,ij->i", elastic, push)[:, None] * push, 0.0)
        return {
            **self.parameters(),
            "friction": friction.tolist(),
            "carried_nodes": [[len(ids) for ids in tool] for tool in carried],
            "clamped_nodes": [[int((clamped[ids] != 0).sum()) for ids in tool] for tool in carried],
            "carried_particles": [[ids.tolist() for ids in tool] for tool in carried],
            "carried_elastic_N": [[elastic[ids].tolist() for ids in tool] for tool in carried],
            "carried_by_law": [[(owner[ids] >= 0).tolist() for ids in tool] for tool in carried],
            "carried_law_part_N": [[returned[ids].tolist() for ids in tool] for tool in carried],
            "jaw_normal_N": [[float(normal[body]) for body in pair] if pair[0] >= 0 else [] for pair in pairs],
            "jaw_closing_direction": [
                [closing[body].tolist() for body in pair] if pair[0] >= 0 else [] for pair in pairs
            ],
            "squeeze_N": squeeze.tolist(),
            "pull_N": pull.tolist(),
            "limit_N": (2.0 * friction * squeeze).tolist(),
            "state": [STATE_NAMES[int(value)] for value in state],
            "clamping_intervals": counters[:, 0].tolist(),
            "sticking_intervals": counters[:, 1].tolist(),
            "slipping_intervals": counters[:, 2].tolist(),
            "slip_length_m": self.slip_length.numpy().astype(float).tolist(),
            "carry_residual_m": self.carry_residual.numpy().astype(float).tolist(),
            "place_travel_m": self.place_travel.numpy().astype(float).tolist(),
            "kept_nodes": int(kept.sum()),
            "kept_particles": np.flatnonzero(kept).tolist(),
            "walking_nodes": int(walking.sum()),
            "walking_particles": np.flatnonzero(walking).tolist(),
            "reaction_N": [[wrench[body, :3].tolist() for body in pair] if pair[0] >= 0 else [] for pair in pairs],
        }
