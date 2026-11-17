# SPDX-License-Identifier: Apache-2.0
"""Exact all-pairs scheduling of the existing chord contact law.

No candidate culling, symmetric-pair substitution, altered reach, or atomics.
Each ordered pair uses the original arithmetic.  A second kernel visits other
indices in the original ascending order, retaining its strict comparisons and
the first equal-distance candidate.  Scratch is bounded before allocation.
"""

import warp as wp

from ..fem.differentiation import active_tape
from .kernels import _node_vector, closest_on_segments


@wp.kernel(enable_backward=False)
def evaluate_pairs(
    q: wp.array[wp.float64],
    nodes: int,
    reach: wp.float64,
    skip: int,
    distances: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.float64],
    points: wp.array2d[wp.vec3d],
    normals: wp.array2d[wp.vec3d],
):
    element, other = wp.tid()
    per_chain = nodes - 1
    chain = element // per_chain
    local = element % per_chain
    distances[element, other] = reach
    if wp.abs(other - local) <= skip:
        return
    first = chain * nodes + local
    a0 = _node_vector(q, first, 0)
    a1 = _node_vector(q, first + 1, 0)
    b0 = _node_vector(q, chain * nodes + other, 0)
    b1 = _node_vector(q, chain * nodes + other + 1, 0)
    pair = closest_on_segments(a0, a1, b0, b1)
    here = a0 + pair[0] * (a1 - a0)
    there = b0 + pair[1] * (b1 - b0)
    distance = wp.length(here - there)
    # This is the original predicate with the initial nearest=reach.  A
    # candidate failing it can never become a later running minimum either.
    if distance < reach and distance > wp.float64(0.0):
        distances[element, other] = distance
        fractions[element, other] = pair[0]
        points[element, other] = wp.float64(0.5) * (here + there)
        normals[element, other] = (here - there) / distance


@wp.kernel(enable_backward=False)
def select_ordered(
    reach: wp.float64,
    distances: wp.array2d[wp.float64],
    fractions: wp.array2d[wp.float64],
    points: wp.array2d[wp.vec3d],
    normals: wp.array2d[wp.vec3d],
    xi: wp.array[wp.float64],
    point: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
):
    element = wp.tid()
    nearest = reach
    normal[element] = wp.vec3d()
    for other in range(distances.shape[1]):
        distance = distances[element, other]
        if distance < nearest and distance > wp.float64(0.0):
            nearest = distance
            xi[element] = fractions[element, other]
            point[element] = points[element, other]
            normal[element] = normals[element, other]


class PairwiseSelfContacts:
    """Fixed owners for capture; bounded small-chain CUDA path."""

    def __init__(self, nodes, n_envs, device, *, max_pairs=131_072, block_dim=128):
        if not isinstance(nodes, int) or not isinstance(n_envs, int) or nodes < 2 or n_envs < 1:
            raise ValueError("Positive chain count and at least two nodes are required")
        shape = (n_envs * (nodes - 1), nodes - 1)
        if shape[0] * shape[1] > max_pairs:
            raise ValueError("Pair scratch exceeds the explicit pair-storage bound")
        self.nodes, self.n_envs = nodes, n_envs
        self.device, self.block_dim = wp.get_device(device), block_dim
        self.distances = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.fractions = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.points = wp.empty(shape, dtype=wp.vec3d, device=self.device)
        self.normals = wp.empty(shape, dtype=wp.vec3d, device=self.device)

    @property
    def storage(self):
        return self.distances, self.fractions, self.points, self.normals

    def find(self, q, reach, skip, xi, point, normal):
        if q.shape != (6 * self.nodes * self.n_envs,) or q.dtype != wp.float64:
            raise ValueError("Coordinate storage does not match the chain layout")
        if any(array.device != self.device or array.requires_grad for array in (q, xi, point, normal)):
            raise ValueError("Pair search requires matching non-gradient storage")
        if active_tape() is not None:
            raise RuntimeError("Use the original thread query under a differentiable tape")
        wp.launch(
            evaluate_pairs,
            dim=self.distances.shape,
            inputs=[q, self.nodes, reach, skip, *self.storage],
            device=self.device,
            block_dim=self.block_dim,
        )
        wp.launch(
            select_ordered,
            dim=self.distances.shape[0],
            inputs=[reach, *self.storage, xi, point, normal],
            device=self.device,
            block_dim=32,
        )
