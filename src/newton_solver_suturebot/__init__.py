# SPDX-License-Identifier: Apache-2.0
"""FEM tissue, rigid needle coupling and SutureBot examples for Newton."""

from ._src.cutting import (
    CutConnectivityDebug,
    CutModel,
    CutSurfaceContact,
    CutTissue,
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
from ._src.solvers import FEMBuilder, LinearFEMData, SolverFEM, SolverLinearFEM
from ._src.solvers.coupled.coupling import NeedleCouplingNotConverged, NeedleTissueCoupler
from ._src.solvers.coupled.interface.insertion import NeedleStepTooLarge, NeedleTissueInteraction
from ._src.solvers.thread import SolverThread, ThreadAttachment, ThreadCapsuleContact, ThreadModel, ThreadTissueCoupling
from ._src.solvers.thread.thread_rigid_contact import ThreadRigidCapsuleContact

__all__ = [
    "CutConnectivityDebug",
    "CutModel",
    "CutSurfaceContact",
    "CutTissue",
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
    "SolverThread",
    "TetrahedralTunnelGuide",
    "ThreadAttachment",
    "ThreadCapsuleContact",
    "ThreadRigidCapsuleContact",
    "ThreadModel",
    "ThreadTissueCoupling",
    "boundary_faces",
    "hex_quadrature",
    "list_mesh_regions",
    "load_mesh",
    "load_mesh_regions",
]
