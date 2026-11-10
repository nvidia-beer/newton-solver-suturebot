# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load an authored rigid needle's capsule geometry and inherited inertial values."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length < 1.0e-12:
        raise ValueError("A finite nonzero direction is required")
    return vector / length


@dataclass(frozen=True)
class NeedleGeometry:
    """Original authored collision arc and inherited demonstration inertia.

    ``centerline`` runs from the modeled collision tip toward the tail.
    ``arc_normal`` makes positive rotation advance the tip away from the shaft.
    The rounded capsule tip is an interaction approximation, not resolved bevel CAD.
    """

    asset_path: Path
    body_path: str
    centerline: np.ndarray
    arc_center: np.ndarray
    arc_normal: np.ndarray
    radius: float
    span: float
    thickness: float
    mass: float
    inertia: np.ndarray

    @property
    def tip(self):
        return self.centerline[0]

    @property
    def tail(self):
        return self.centerline[-1]

    def sample_centerline(self, spacing):
        """Sample the unchanged piecewise capsule chain, retaining every endpoint."""
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Needle shaft sampling spacing must be finite and positive")
        segments = []
        for start, end in zip(self.centerline[:-1], self.centerline[1:], strict=True):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
            segments.append(start + (np.arange(count) / count)[:, None] * (end - start))
        return np.concatenate((*segments, self.centerline[-1:]))

    @classmethod
    def load(cls, asset_path=None):
        """Read the extracted USD; fit and validate its unchanged capsule endpoints."""
        from pxr import Usd, UsdGeom

        path = Path(asset_path or Path(__file__).resolve().parents[2] / "examples/assets/needle_insertion/needle.usda")
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError(f"Cannot open needle USD: {path}")
        body = stage.GetDefaultPrim()
        segments, radii = [], []
        for prim in sorted(body.GetChildren(), key=lambda p: p.GetName()):
            if prim.GetTypeName() != "Capsule":
                continue
            axis = str(prim.GetAttribute("axis").Get())
            height = float(prim.GetAttribute("height").Get())
            local = np.zeros((2, 3))
            local[:, "XYZ".index(axis)] = (-height / 2, height / 2)
            transform = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())
            segments.append(local @ transform[:3, :3] + transform[3, :3])
            radii.append(float(prim.GetAttribute("radius").Get()))
        if len(segments) < 3:
            raise ValueError("Needle asset needs an authored chain of at least three capsules")
        segments = np.asarray(segments)
        if np.max(np.linalg.norm(segments[:-1, 1] - segments[1:, 0], axis=1)) > 1.0e-7:
            raise ValueError("The authored needle capsules do not form one connected ordered arc")
        points = np.concatenate((segments[:, 0], segments[-1:, 1]))
        mean = points.mean(axis=0)
        _, _, basis = np.linalg.svd(points - mean)
        planar = (points - mean) @ basis[:2].T
        solution = np.linalg.lstsq(np.c_[2 * planar, np.ones(len(points))], (planar * planar).sum(axis=1), rcond=None)[
            0
        ]
        center = mean + solution[:2] @ basis[:2]
        radius = float(np.sqrt(solution[2] + solution[:2] @ solution[:2]))
        radial = points - center
        normal = _unit(np.cross(radial[1], radial[0]))
        u, v = _unit(radial[0]), _unit(np.cross(normal, radial[0]))
        angles = np.unwrap(np.arctan2(radial @ v, radial @ u))
        residual = max(np.max(np.abs(np.linalg.norm(radial, axis=1) - radius)), np.max(np.abs(radial @ normal)))
        if residual > 1.0e-6 or not np.all(np.diff(angles) < 0) or not 0 < -angles[-1] < np.pi:
            raise ValueError("The authored collision needle is not a supported planar minor circular arc")
        if not np.allclose(radii, radii[0], rtol=0, atol=1.0e-9):
            raise ValueError("The authored capsule radii are inconsistent")
        mass = float(body.GetAttribute("physics:mass").Get())
        inertia = np.asarray(body.GetAttribute("physics:diagonalInertia").Get(), dtype=float)
        if mass <= 0 or inertia.shape != (3,) or np.any(inertia <= 0):
            raise ValueError("The needle requires positive authored mass and diagonal inertia")
        return cls(
            path.resolve(),
            str(body.GetPath()),
            points,
            center,
            normal,
            radius,
            float(-angles[-1]),
            radii[0],
            mass,
            inertia,
        )
