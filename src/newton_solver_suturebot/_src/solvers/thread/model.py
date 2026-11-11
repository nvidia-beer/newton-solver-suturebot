# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference configuration of a thread: one chain of two-node cubic Hermite cable elements."""

from dataclasses import dataclass

import numpy as np
import warp as wp


def _shape_functions(xi: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite shape values and first derivatives at ``xi`` in [0, 1] of an element of ``length``.

    Entries alternate position and slope: [node a, slope a, node b, slope b].
    """
    x2, x3 = xi * xi, xi * xi * xi
    shape = np.array(
        [1.0 - 3.0 * x2 + 2.0 * x3, length * (xi - 2.0 * x2 + x3), 3.0 * x2 - 2.0 * x3, length * (-x2 + x3)]
    )
    first = np.array(
        [
            (6.0 * x2 - 6.0 * xi) / length,
            1.0 - 4.0 * xi + 3.0 * x2,
            (6.0 * xi - 6.0 * x2) / length,
            -2.0 * xi + 3.0 * x2,
        ]
    )
    return shape, first


@dataclass
class ThreadModel:
    """One stress-free chain of cable elements with a round, uniform section.

    Every node has six coordinates: its position [m] and its slope, the
    derivative of the centerline along the reference length [dimensionless].
    Element ``i`` joins nodes ``i`` and ``i + 1``. The elements resist
    stretching and bending; they have no twist.

    Attributes:
        rest_q: Reference coordinates, ``6 * node_count`` values.
        lengths: Reference length of each element [m].
        axial_stiffness: E A of each element [N].
        bending_stiffness: E I of each element [N m²].
        mass_diagonal: Consistent mass of each node as a 2-by-2 block over
            (position, slope); every block acts on the three axes alike.
        mass_coupling: Mass block between node ``i`` (rows) and node
            ``i + 1`` (columns).
        node_count: Number of nodes.
        device: Warp device of the arrays.
    """

    rest_q: wp.array
    lengths: wp.array
    axial_stiffness: wp.array
    bending_stiffness: wp.array
    mass_diagonal: wp.array
    mass_coupling: wp.array
    node_count: int
    device: wp.Device

    @property
    def element_count(self) -> int:
        return self.node_count - 1

    @property
    def dof_count(self) -> int:
        return 6 * self.node_count

    @staticmethod
    def create(
        positions: np.ndarray,
        *,
        young_modulus: float,
        radius: float,
        density: float,
        device: wp.DeviceLike = None,
    ) -> "ThreadModel":
        """Build a chain through ``positions`` that is stress-free in that shape.

        Args:
            positions: Node positions [m], shape (node_count, 3), at least three
                nodes. The slope at an inner node is the unit bisector of its
                two chords; at an end it is the unit chord.
            young_modulus: Young's modulus [Pa].
            radius: Section radius [m].
            density: Material density [kg/m³].
            device: Warp device; defaults to the current device.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 3:
            raise ValueError("positions must have shape (node_count >= 3, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if not all(np.isfinite(value) and value > 0.0 for value in (young_modulus, radius, density)):
            raise ValueError("young_modulus, radius and density must be finite and positive")
        chords = np.diff(positions, axis=0)
        lengths = np.linalg.norm(chords, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("neighbouring nodes must be distinct")
        directions = chords / lengths[:, None]
        slopes = np.zeros_like(positions)
        slopes[0], slopes[-1] = directions[0], directions[-1]
        slopes[1:-1] = directions[:-1] + directions[1:]
        norms = np.linalg.norm(slopes, axis=1)
        if np.any(norms < 1.0e-12):
            raise ValueError("the chain must not reverse at a node")
        slopes /= norms[:, None]

        nodes = len(positions)
        area = np.pi * radius**2
        diagonal = np.zeros((nodes, 2, 2))
        coupling = np.zeros((nodes - 1, 2, 2))
        gauss_points, gauss_weights = np.polynomial.legendre.leggauss(5)
        for element, length in enumerate(lengths):
            mass = np.zeros((4, 4))
            for point, weight in zip(gauss_points, gauss_weights, strict=True):
                shape, first = _shape_functions(0.5 * (point + 1.0), length)
                tangent = (
                    first[2] * (positions[element + 1] - positions[element])
                    + first[1] * slopes[element]
                    + first[3] * slopes[element + 1]
                )
                # The mass is integrated along the reference curve, not along its chord.
                mass += np.outer(shape, shape) * (0.5 * weight * length * np.linalg.norm(tangent) * density * area)
            diagonal[element] += mass[:2, :2]
            diagonal[element + 1] += mass[2:, 2:]
            coupling[element] = mass[:2, 2:]

        target = wp.get_device(device)

        def array(values, dtype=wp.float64):
            return wp.array(values, dtype=dtype, device=target)

        return ThreadModel(
            rest_q=array(np.column_stack((positions, slopes)).reshape(-1)),
            lengths=array(lengths),
            axial_stiffness=array(np.full(nodes - 1, young_modulus * area)),
            bending_stiffness=array(np.full(nodes - 1, young_modulus * (0.25 * np.pi * radius**4))),
            mass_diagonal=array(diagonal, wp.mat22d),
            mass_coupling=array(coupling, wp.mat22d),
            node_count=nodes,
            device=target,
        )
