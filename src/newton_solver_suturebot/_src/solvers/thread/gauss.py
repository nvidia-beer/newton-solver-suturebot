# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Parallel quadrature for small batches, retaining the five-point force law."""

import warp as wp

from .kernels import (
    _centerline_derivatives,
    _curvature_coefficients,
    _gauss_point,
    _gauss_weight,
    _shape_derivatives,
)


@wp.kernel(enable_backward=False)
def evaluate_gauss_points(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    axial_stiffness: wp.array[wp.float64],
    bending_stiffness: wp.array[wp.float64],
    rest_q: wp.array[wp.float64],
    tangent_blocks: bool,
    status: wp.array[int],
    gradient: wp.array3d[wp.vec3d],
    hessian: wp.array4d[wp.mat33d],
):
    """The unchanged element integrand, independently at each of five Gauss points."""
    index, point = wp.tid()
    per_chain = lengths.shape[0]
    if status[index // per_chain] != 0:
        return
    element = index % per_chain
    node_a = (index // per_chain) * (per_chain + 1) + element
    node_b = node_a + 1
    length = lengths[element]
    axial, bending = axial_stiffness[element], bending_stiffness[element]
    zero = wp.float64(0.0)
    two = wp.float64(2.0)
    identity = wp.identity(n=3, dtype=wp.float64)
    first, second = _shape_derivatives(_gauss_point(point), length)
    tangent, curvature = _centerline_derivatives(q, node_a, node_b, first, second)
    rest_a, rest_b = _centerline_derivatives(rest_q, element, element + 1, first, second)
    rest_metric = wp.dot(rest_a, rest_a)
    metric = wp.dot(tangent, tangent) + wp.float64(1.0e-24)
    inv_metric = wp.float64(1.0) / metric
    inv_cube = inv_metric * inv_metric * inv_metric
    strain = wp.float64(0.5) * (wp.dot(tangent, tangent) / rest_metric - wp.float64(1.0))
    cross = wp.cross(tangent, curvature)
    numerator = wp.dot(cross, cross)
    _, bending_first, bending_second = _curvature_coefficients(tangent, curvature, rest_a, rest_b, length)
    weight = length * _gauss_weight(point) * wp.sqrt(rest_metric)
    skew_tangent, skew_curvature = wp.skew(tangent), wp.skew(curvature)
    vector_scale = inv_metric * wp.sqrt(inv_metric)
    tension = wp.max(strain, zero)
    stiffening = wp.max(bending_first, zero)
    # Shared by every nodal vector: d(numerator) and d(k) along r' and along r''.
    numerator_first = two * wp.cross(curvature, cross)
    numerator_second = two * wp.cross(cross, tangent)
    vector_first = vector_scale * (-skew_curvature - wp.float64(3.0) * inv_metric * wp.outer(cross, tangent))
    vector_second = vector_scale * skew_tangent
    for b in range(4):
        # Along coordinate e of vector b: d(r') = first[b] e and d(r'') = second[b] e.
        strain_b = (first[b] / rest_metric) * tangent
        numerator_b = first[b] * numerator_first + second[b] * numerator_second
        curvature_b = inv_cube * (numerator_b - wp.float64(6.0) * numerator * inv_metric * first[b] * tangent)
        value = weight * (axial * strain * strain_b + bending * bending_first * curvature_b)
        gradient[index, point, b] = value
        if tangent_blocks:
            vector_b = first[b] * vector_first + second[b] * vector_second
            for c in range(b, 4):
                strain_c = (first[c] / rest_metric) * tangent
                numerator_c = first[c] * numerator_first + second[c] * numerator_second
                curvature_c = inv_cube * (numerator_c - wp.float64(6.0) * numerator * inv_metric * first[c] * tangent)
                vector_c = first[c] * vector_first + second[c] * vector_second
                block = weight * (
                    axial * (wp.outer(strain_b, strain_c) + tension * (first[b] * first[c] / rest_metric) * identity)
                    + bending
                    * (
                        bending_second * wp.outer(curvature_b, curvature_c)
                        + two * stiffening * wp.transpose(vector_b) * vector_c
                    )
                )
                hessian[index, point, b, c] = block


@wp.kernel(enable_backward=False)
def reduce_gauss_points(
    partial_gradient: wp.array3d[wp.vec3d],
    partial_hessian: wp.array4d[wp.mat33d],
    elements_per_chain: int,
    tangent_blocks: bool,
    status: wp.array[int],
    gradient: wp.array2d[wp.vec3d],
    hessian: wp.array3d[wp.mat33d],
):
    element, b = wp.tid()
    if status[element // elements_per_chain] != 0:
        return
    force = wp.vec3d()
    # Match the original quadrature order rather than an atomic/tree reduction.
    for point in range(5):
        force += partial_gradient[element, point, b]
    gradient[element, b] = force
    if tangent_blocks:
        for c in range(4):
            block = wp.mat33d()
            low, high = wp.min(b, c), wp.max(b, c)
            for point in range(5):
                block += partial_hessian[element, point, low, high]
            if c < b:
                block = wp.transpose(block)
            hessian[element, b, c] = block


class ParallelGaussElements:
    """Persistent partial integrands; no forces, quadrature or budgets changed."""

    def __init__(self, thread, count, *, block_dim=8):
        self.thread, self.count, self.block_dim = thread, count, block_dim
        self.gradient = wp.empty((count, 5, 4), dtype=wp.vec3d, device=thread.device)
        self.hessian = wp.empty((count, 5, 4, 4), dtype=wp.mat33d, device=thread.device)

    def evaluate(self, q, tangent_blocks, status, gradient, hessian):
        thread = self.thread
        wp.launch(
            evaluate_gauss_points,
            (self.count, 5),
            [
                q,
                thread.lengths,
                thread.axial_stiffness,
                thread.bending_stiffness,
                thread.rest_q,
                tangent_blocks,
                status,
                self.gradient,
                self.hessian,
            ],
            device=thread.device,
            block_dim=self.block_dim,
        )
        wp.launch(
            reduce_gauss_points,
            (self.count, 4),
            [self.gradient, self.hessian, thread.element_count, tangent_blocks, status, gradient, hessian],
            device=thread.device,
            block_dim=self.block_dim,
        )
