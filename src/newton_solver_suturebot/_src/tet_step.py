# SPDX-License-Identifier: Apache-2.0
"""Reject unresolved Tet4 motion before committing a coupled physical step.

Positions are never projected. A rejected step must be recomputed from its
unchanged start with a smaller timestep. Along affine nodal motion the signed
volume is cubic, so positive endpoints alone do not establish feasibility.
"""

import math

import warp as wp

from .insertion import NeedleStepTooLarge


@wp.func
def _value(coefficients: wp.vec4d, t: wp.float64):
    return ((coefficients[3] * t + coefficients[2]) * t + coefficients[1]) * t + coefficients[0]


@wp.func
def _segment_minimum(coefficients: wp.vec4d):
    result = wp.min(coefficients[0], _value(coefficients, wp.float64(1.0)))
    # Solve the quadratic derivative with the stable q-formula; it preserves
    # the finite root even when the cubic coefficient is nearly zero.
    a, b, c = wp.float64(3.0) * coefficients[3], wp.float64(2.0) * coefficients[2], coefficients[1]
    # Finite volume coefficients can still overflow when the discriminant
    # squares them. Scaling leaves the roots unchanged and bounds that work.
    scale = wp.max(wp.abs(a), wp.max(wp.abs(b), wp.abs(c)))
    if scale > wp.float64(0.0):
        a, b, c = a / scale, b / scale, c / scale
    if a != wp.float64(0.0):
        discriminant = b * b - wp.float64(4.0) * a * c
        if discriminant >= wp.float64(0.0):
            root = wp.sqrt(discriminant)
            q = -wp.float64(0.5) * (b + root)
            if b < wp.float64(0.0):
                q = -wp.float64(0.5) * (b - root)
            if q != wp.float64(0.0):
                first, second = q / a, c / q
                if first > wp.float64(0.0) and first < wp.float64(1.0):
                    result = wp.min(result, _value(coefficients, first))
                if second > wp.float64(0.0) and second < wp.float64(1.0):
                    result = wp.min(result, _value(coefficients, second))
    elif b != wp.float64(0.0):
        root = -c / b
        if root > wp.float64(0.0) and root < wp.float64(1.0):
            result = wp.min(result, _value(coefficients, root))
    return result


@wp.kernel(enable_backward=False)
def _minimum_tet_motion(
    start: wp.array[wp.vec3],
    end: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    result: wp.array[wp.float64],
):
    element = wp.tid()
    # Same activation contract as eval_tetrahedra: reserved cut slots carry
    # no material and may have unset virtual nodes/rest geometry.
    if materials[element, 0] == 0.0 and materials[element, 1] == 0.0 and materials[element, 2] == 0.0:
        return
    n0, n1, n2, n3 = indices[element, 0], indices[element, 1], indices[element, 2], indices[element, 3]
    origin = wp.vec3d(start[n0])
    a, b, c = wp.vec3d(start[n1]) - origin, wp.vec3d(start[n2]) - origin, wp.vec3d(start[n3]) - origin
    origin = wp.vec3d(end[n0])
    da, db, dc = wp.vec3d(end[n1]) - origin - a, wp.vec3d(end[n2]) - origin - b, wp.vec3d(end[n3]) - origin - c
    inverse_volume = wp.determinant(wp.mat33d(poses[element]))
    coefficients = (
        wp.vec4d(
            wp.dot(a, wp.cross(b, c)),
            wp.dot(da, wp.cross(b, c)) + wp.dot(a, wp.cross(db, c)) + wp.dot(a, wp.cross(b, dc)),
            wp.dot(a, wp.cross(db, dc)) + wp.dot(da, wp.cross(b, dc)) + wp.dot(da, wp.cross(db, c)),
            wp.dot(da, wp.cross(db, dc)),
        )
        * inverse_volume
    )
    scale = (wp.length(a) + wp.length(da)) * (wp.length(b) + wp.length(db)) * (wp.length(c) + wp.length(dc))
    scale *= inverse_volume
    valid = wp.isfinite(inverse_volume) and inverse_volume > wp.float64(0.0) and wp.isfinite(scale)
    for component in range(4):
        valid = valid and wp.isfinite(coefficients[component])
    if not valid:
        wp.atomic_min(result, 0, wp.float64(-wp.inf))
        return
    # Avoid accepting a numerically indistinguishable zero by reserving a small
    # float64 evaluation margin, scaled by geometry rather than canceling
    # polynomial coefficients. This is a feasibility margin, not a material law.
    margin = wp.float64(64.0 * 2.220446049250313e-16) * scale
    wp.atomic_min(result, 0, _segment_minimum(coefficients) - margin)


class TetStepGuard:
    """Check unchanged model Tet4 topology with one compact host readback."""

    def __init__(self, model):
        self.model = model
        self._minimum = wp.empty(1, dtype=wp.float64, device=model.device)

    def check(self, state_in, state_out):
        if not self.model.tet_count:
            return math.inf
        self._minimum.fill_(math.inf)
        wp.launch(
            _minimum_tet_motion,
            self.model.tet_count,
            [
                state_in.particle_q,
                state_out.particle_q,
                self.model.tet_indices,
                self.model.tet_poses,
                self.model.tet_materials,
                self._minimum,
            ],
            device=self.model.device,
        )
        minimum = float(self._minimum.numpy()[0])
        if math.isnan(minimum) or minimum <= 0:
            raise NeedleStepTooLarge(
                f"Unresolved FEM tetrahedron motion: nonpositive swept volume bound J={minimum:.6g}; "
                "retry from the accepted state with a smaller timestep"
            )
        return minimum
