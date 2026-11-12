# SPDX-License-Identifier: Apache-2.0
"""PSD search tangent for a frozen surface facet; never a physical force law.

The stiffness term uses an endpoint displacement approximation h*delta_velocity.
Facet geometry, normalized weights, cap choice, and the normal load in the
frozen-friction quadratic are frozen. Sliding radial curvature is retained
as a PSD majorizer of the frozen-load friction potential, not its exact
Hessian. Geometry/normalization derivatives and the
nonsymmetric normal/friction cross terms are intentionally omitted. Exact
original surface forces and coupling acceptance must remain authoritative.
"""

import math

import warp as wp


@wp.func
def surface_contact_tangent(
    weight: float,
    penetration: float,
    normal_speed: float,
    normal_force: float,
    tangent: wp.vec3,
    normal: wp.vec3,
    stiffness: float,
    damping: float,
    radius: float,
    friction: float,
    speed_scale: float,
    dt: float,
):
    normal_coefficient = weight * dt * (stiffness - damping / radius * wp.min(normal_speed, 0.0))
    if normal_speed < 0.0:
        normal_coefficient += weight * damping * penetration / radius
    result = normal_coefficient * wp.outer(normal, normal)
    projection = wp.identity(n=3, dtype=float) - wp.outer(normal, normal)
    speed = wp.length(tangent)
    # Frozen-coefficient PSD approximation retains the radial direction;
    # the exact sliding derivative removes it and can overshoot a slip reversal.
    # This changes the search metric only, never the original physical force.
    result += friction * normal_force / wp.max(speed, speed_scale) * projection
    return result


class SurfaceTangentAnchors:
    """Bounded scratch for all contributing sample/facet pairs.

    Count can exceed capacity. A consumer must reject this approximation if
    either flag is set: [overflow, impulse-capped force mode]. Original physical
    forces still include every facet even when recording cannot be used.
    Clear before each fresh candidate; retain unchanged for dual-only retries.
    """

    def __init__(self, capacity, device="cpu", *, enabled=True):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        self.count = wp.zeros(1 if enabled else 0, dtype=int, device=device)
        self.flags = wp.zeros(2, dtype=int, device=device)
        self.nodes = wp.empty(capacity, dtype=wp.vec3i, device=device)
        self.bary = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.points = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.levers = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.normals = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.tangents = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.matrices = wp.empty(capacity, dtype=wp.mat33, device=device)
        self.sources = wp.empty(capacity, dtype=wp.vec2i, device=device)
        # a, penetration, vn, fn, radius, normalized facet weight, cap limit.
        self.scalars = wp.empty((capacity, 7), dtype=float, device=device)

    def clear(self):
        self.count.zero_()
        self.flags.zero_()

    def arguments(self, moment_length):
        if not math.isfinite(moment_length) or moment_length <= 0:
            raise ValueError("moment_length must be positive")
        return [
            float(moment_length),
            self.count,
            self.flags,
            self.nodes,
            self.bary,
            self.points,
            self.levers,
            self.normals,
            self.tangents,
            self.matrices,
            self.sources,
            self.scalars,
        ]
