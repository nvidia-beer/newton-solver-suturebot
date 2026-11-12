# SPDX-License-Identifier: Apache-2.0
"""One complete exterior pad surface with live active-region vertex updates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _positions(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) == 0
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain finite nonempty [V,3] coordinates")
    return np.array(array, dtype=np.result_type(array.dtype, np.float32), copy=True)


def _triangles(values: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain nonempty integer [F,3] indices")
    if np.any(array < 0) or np.any(array >= count):
        raise ValueError(f"{name} contains out-of-range vertex indices")
    array = array.astype(np.int64, copy=True)
    sorted_faces = np.sort(array, axis=1)
    if np.any(np.diff(sorted_faces, axis=1) == 0):
        raise ValueError(f"{name} contains repeated vertices in a triangle")
    if len(np.unique(sorted_faces, axis=0)) != len(array):
        raise ValueError(f"{name} contains duplicate exterior triangles")
    return array


def _exact_coordinate_mapping(active: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match legacy coordinates exactly; no nearest-point geometric guesses."""
    dtype = np.result_type(active.dtype, reference.dtype)
    key_dtype = np.dtype([("x", dtype), ("y", dtype), ("z", dtype)])
    reference_keys = np.ascontiguousarray(reference, dtype=dtype).view(key_dtype).reshape(-1)
    active_keys = np.ascontiguousarray(active, dtype=dtype).view(key_dtype).reshape(-1)
    keys, first, counts = np.unique(reference_keys, return_index=True, return_counts=True)
    locations = np.searchsorted(keys, active_keys)
    if np.any(locations == len(keys)) or np.any(keys[np.minimum(locations, len(keys) - 1)] != active_keys):
        raise ValueError(
            "Legacy active vertices require exact reference coordinate matches; export active_reference_indices"
        )
    if np.any(counts[locations] != 1):
        raise ValueError("Ambiguous legacy reference coordinates; export explicit active_reference_indices")
    result = first[locations]
    if len(np.unique(result)) != len(result):
        raise ValueError("Ambiguous duplicate active coordinates; export explicit active_reference_indices")
    return result.astype(np.int64)


def _explicit_mapping(values: Any, active: np.ndarray, reference: np.ndarray) -> np.ndarray:
    indices = np.asarray(values)
    if indices.shape != (len(active),) or indices.dtype.kind not in "iu":
        raise ValueError("active_reference_indices must contain one integer index per active vertex")
    if np.any(indices < 0) or np.any(indices >= len(reference)):
        raise ValueError("active_reference_indices contains out-of-range reference vertices")
    indices = indices.astype(np.int64, copy=True)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("active_reference_indices must be one-to-one")
    # IDs carry the binding. This check catches a wrong order/asset while
    # permitting the representational rounding of float32 versus float64.
    epsilon = max(np.finfo(active.dtype).eps, np.finfo(reference.dtype).eps)
    scale = max(float(np.abs(active).max()), float(np.abs(reference[indices]).max()), 1e-6)
    if not np.allclose(active, reference[indices], rtol=0, atol=4 * epsilon * scale):
        raise ValueError("active_reference_indices does not match the active rest geometry")
    return indices


class FullSurfaceGeometry:
    """Compact a single exterior mesh and bind its active physical particles.

    ``active_reference_indices[i]`` identifies active particle ``i`` in the
    complete reference vertex array. Interior reference vertices are discarded
    from the render mesh. Shared surface vertices occur once, so the active
    tissue is never overlaid by an undeformed duplicate surface.

    Legacy geometry without exported indices can use exact, unique coordinate
    matches. Generic scenes with no complete reference arrays use their tissue
    mesh directly. Arrays exposed by this class are independent, read-only
    copies; ``update`` returns a fresh batch and never changes its inputs.
    """

    def __init__(self, geometry: dict[str, Any]) -> None:
        active = _positions(geometry["tissue_rest_positions"], "tissue_rest_positions")
        have_reference = "reference_positions" in geometry
        if have_reference != ("reference_triangles" in geometry):
            raise ValueError("Complete reference_positions and reference_triangles must be provided together")
        self.active_vertex_count = len(active)
        if have_reference:
            reference = _positions(geometry["reference_positions"], "reference_positions")
            faces = _triangles(geometry["reference_triangles"], len(reference), "reference_triangles")
            if "active_reference_indices" in geometry:
                mapping = _explicit_mapping(geometry["active_reference_indices"], active, reference)
                self.mapping_provenance = "exported active_reference_indices"
            else:
                mapping = _exact_coordinate_mapping(active, reference)
                self.mapping_provenance = "legacy exact unique coordinate match"
        else:
            if "active_reference_indices" in geometry:
                raise ValueError("active_reference_indices requires the complete reference geometry")
            reference = active
            faces = _triangles(geometry["tissue_triangles"], len(reference), "tissue_triangles")
            mapping = np.arange(len(active), dtype=np.int64)
            self.mapping_provenance = "generic tissue surface"
        self.reference_vertex_count = len(reference)
        self.reference_vertex_indices, compact = np.unique(faces.reshape(-1), return_inverse=True)
        self.rest_positions = reference[self.reference_vertex_indices].copy()
        self.triangles = compact.reshape(-1, 3).astype(np.int32)
        surface_index = np.full(len(reference), -1, dtype=np.int64)
        surface_index[self.reference_vertex_indices] = np.arange(len(self.reference_vertex_indices))
        active_surface = surface_index[mapping]
        visible = active_surface >= 0
        self.active_particle_indices = np.flatnonzero(visible)
        self.active_surface_indices = active_surface[visible]
        for array in (
            self.reference_vertex_indices,
            self.rest_positions,
            self.triangles,
            self.active_particle_indices,
            self.active_surface_indices,
        ):
            array.flags.writeable = False

    def update(self, positions: Any) -> np.ndarray:
        """Return ``[N,S,3]`` exterior vertices from live ``[N,V,3]`` particles.

        Static vertices retain their exact stored reference values. Output
        precision can increase to preserve the supplied active coordinates;
        any float32 conversion for rendering belongs to the caller.
        """
        positions = np.asarray(positions)
        if (
            positions.ndim != 3
            or positions.shape[0] < 1
            or positions.shape[1:] != (self.active_vertex_count, 3)
            or positions.dtype.kind not in "fiu"
            or not np.isfinite(positions).all()
        ):
            raise ValueError(f"Active positions must be finite [N,{self.active_vertex_count},3] with N>=1")
        dtype = np.result_type(self.rest_positions.dtype, positions.dtype)
        output = np.broadcast_to(self.rest_positions, (len(positions), *self.rest_positions.shape)).astype(
            dtype, copy=True
        )
        output[:, self.active_surface_indices] = positions[:, self.active_particle_indices]
        return output
