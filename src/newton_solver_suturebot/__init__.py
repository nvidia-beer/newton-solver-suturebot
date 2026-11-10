# SPDX-License-Identifier: Apache-2.0
"""FEM tissue, rigid needle coupling and SutureBot examples for Newton."""

from ._src.coupling import NeedleCouplingNotConverged, NeedleTissueCoupler
from ._src.cutting import (
    CutConnectivityDebug,
    CutModel,
    CutSurfaceContact,
    CutVolumeGuard,
    LinearTetCut,
    MaterialHolds,
    TetrahedralTunnelGuide,
)
from ._src.geometry import (
    HexMesh,
    Mesh,
    MeshRegion,
    NeedleGeometry,
    RegionMesh,
    boundary_faces,
    hex_quadrature,
    list_mesh_regions,
    load_mesh,
    load_mesh_regions,
)
from ._src.insertion import NeedleStepTooLarge, NeedleTissueInteraction
from ._src.solvers import FEMBuilder, LinearFEMData, SolverFEM, SolverLinearFEM
from ._src.solvers.mujoco_fem.solver import SolverMuJoCoFEM

__all__ = [
    "CutConnectivityDebug",
    "CutModel",
    "CutSurfaceContact",
    "CutVolumeGuard",
    "FEMBuilder",
    "HexMesh",
    "LinearFEMData",
    "LinearTetCut",
    "MaterialHolds",
    "Mesh",
    "MeshRegion",
    "NeedleCouplingNotConverged",
    "NeedleGeometry",
    "NeedleStepTooLarge",
    "NeedleTissueCoupler",
    "NeedleTissueInteraction",
    "RegionMesh",
    "SolverFEM",
    "SolverLinearFEM",
    "SolverMuJoCoFEM",
    "TetrahedralTunnelGuide",
    "boundary_faces",
    "hex_quadrature",
    "list_mesh_regions",
    "load_mesh",
    "load_mesh_regions",
]
