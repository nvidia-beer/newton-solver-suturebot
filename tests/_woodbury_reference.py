"""Independent dense contact transpose oracle for the small physical fixture."""

import numpy as np


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def interpolation(fixed, nodes, bary):
    """The full interpolation retains fixed support reactions in its transpose."""
    fixed = np.asarray(fixed, dtype=bool)
    nodes, bary = np.asarray(nodes), np.asarray(bary, dtype=float)
    if nodes.shape != bary.shape or nodes.ndim != 2 or nodes.shape[1] != 4:
        raise ValueError("Expected tetrahedron node/barycentric arrays")
    if np.any(nodes < 0) or np.any(nodes >= len(fixed)) or not np.isfinite(bary).all():
        raise ValueError("Invalid material interpolation")
    full = np.zeros((3 * len(nodes), 3 * len(fixed)))
    for anchor, (indices, weights) in enumerate(zip(nodes, bary, strict=True)):
        for node, weight in zip(indices, weights, strict=True):
            full[3 * anchor : 3 * anchor + 3, 3 * node : 3 * node + 3] += weight * np.eye(3)
    free = np.flatnonzero(np.repeat(~fixed, 3))
    return full, full[:, free], free


def rigid_interpolation(levers):
    return np.vstack([np.column_stack((np.eye(3), -skew(lever))) for lever in levers])


def block_diagonal(blocks):
    result = np.zeros((3 * len(blocks), 3 * len(blocks)))
    for index, block in enumerate(blocks):
        result[3 * index : 3 * index + 3, 3 * index : 3 * index + 3] = block
    return result


def sliding_shears(matrices, friction):
    """Shear of every sliding tangent of a friction tunnel; zero for every other tangent.

    A sliding anchor has the tangent a ((1 - f) (I - t t.T) + f d d.T) of rank two: its null vector is the
    shaft tangent t and the eigenvector of its largest eigenvalue the unit direction d of the wall
    reaction. Its shear is friction * sign * t d.T; the sign of the drag alternates along the track.
    """
    matrices = np.asarray(matrices)
    shears = np.zeros_like(matrices)
    for index, matrix in enumerate(matrices.astype(float)):
        if np.linalg.matrix_rank(matrix, tol=1.0e-5) != 2:
            continue
        _, vectors = np.linalg.eigh(matrix)
        shears[index] = friction * (-1.0) ** index * np.outer(vectors[:, 0], vectors[:, 2])
    return shears


def sheared(matrices, shears):
    """The anchor tangents (I + S) D as the operator applies them, in FP64."""
    return [
        (np.eye(3) + shear) @ matrix for shear, matrix in zip(np.asarray(shears, dtype=float), matrices, strict=True)
    ]


def nonzero_columns(factors):
    """Rows 3 * track + column of a padded Schur system whose tangent-factor column is not zero, in order."""
    return [
        3 * track + column
        for track, factor in enumerate(np.asarray(factors))
        for column in range(3)
        if factor[:, column].any()
    ]


def compacted_solution(matrix, rhs, rows):
    """FP64 solution of a padded Schur system from its rows ``rows`` alone; every other unknown is zero."""
    matrix, rhs = np.asarray(matrix, dtype=float), np.asarray(rhs, dtype=float)
    solution = np.zeros(len(rhs))
    solution[rows] = np.linalg.solve(matrix[np.ix_(rows, rows)], rhs[rows])
    return solution
