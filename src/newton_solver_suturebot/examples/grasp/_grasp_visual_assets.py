# SPDX-License-Identifier: Apache-2.0
"""Read original bundled dVRK visual assets without replaying recorded motion.

The prepared phantom has authored constant materials, not photographic textures.
Distal tool parts are assembled from USD joint frames at explicitly chosen joint
coordinates. These are render assets, not a grasp or articulated-physics model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..recording._suturebot_recording import DEFAULT_RECORDING


@dataclass(frozen=True)
class VisualMesh:
    """A triangle mesh in the declared frame, with source material properties."""

    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    source_vertex_indices: np.ndarray
    diffuse_rgb: tuple[float, float, float]
    roughness: float
    metallic: float
    opacity: float
    uvs: np.ndarray | None
    texture_path: Path | None
    source_prim: str
    coordinate_frame: str
    normals: np.ndarray | None = None

    def to_newton_mesh(self):
        """Build a render-only Newton mesh; no geometry or inertia approximation."""
        import newton

        return newton.Mesh(
            self.vertices,
            self.triangles.reshape(-1),
            normals=self.normals,
            uvs=self.uvs,
            color=self.diffuse_rgb,
            roughness=self.roughness,
            metallic=self.metallic,
            opacity=self.opacity,
            texture=None if self.texture_path is None else str(self.texture_path),
            compute_inertia=False,
        )


@dataclass(frozen=True)
class VisualAsset:
    meshes: tuple[VisualMesh, ...]
    metadata: dict[str, Any]
    component_transforms: dict[str, np.ndarray]


def physical_tool_parts(asset, assembly, *, body_count):
    """Bind CAD to actual tool links, preserving the older distal-body layout.

    The loader assembles all meshes in its zero-angle wrist-yaw frame, including
    shaft shortening. An articulated tool supplies ``part_body_indices``; those
    meshes are transformed back to their component-local frames before their
    independently solved body poses are applied. No recorded pose is used.
    """
    mapping = assembly.get("part_body_indices")
    if mapping is not None and not isinstance(mapping, dict):
        raise ValueError("Articulated tool parts require a component-to-body mapping")
    result = []
    for mesh in asset.meshes:
        if mapping is None:
            body = assembly["palm_body_index"]
            if mesh.name in ("jaw1", "jaw2"):
                body = assembly["jaw_body_indices"][int(mesh.name[-1]) - 1]
        else:
            if mesh.name not in mapping or mesh.name not in asset.component_transforms:
                raise ValueError(f"Articulated tool is missing the body or source frame for {mesh.name}")
            body = mapping[mesh.name]
            transform = _transform(asset.component_transforms[mesh.name])
            # Invert the proper rigid transform without changing topology,
            # materials, texture coordinates or the shortened shaft profile.
            vertices = (mesh.vertices - transform[:3, 3]) @ transform[:3, :3]
            normals = None if mesh.normals is None else mesh.normals @ transform[:3, :3]
            mesh = replace(
                mesh,
                vertices=vertices.astype(np.float32),
                normals=None if normals is None else normals.astype(np.float32),
                coordinate_frame=f"{assembly['instrument']} {mesh.name} body",
            )
        if isinstance(body, (bool, np.bool_)) or not isinstance(body, (int, np.integer)) or not 0 <= body < body_count:
            raise ValueError("Visual tool part refers to an absent physical body")
        result.append((int(body), mesh))
    return result


def _recording(recording):
    path = DEFAULT_RECORDING if recording is None else Path(getattr(recording, "path", recording))
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text())
    root = (path.parent / manifest["asset_root"]).resolve()
    scene = (path.parent / manifest["scene_usd"]).resolve()
    if not scene.is_relative_to(root) or not scene.is_file():
        raise ValueError("The visual scene must be an existing bundled asset")
    return path, manifest, root, scene


def _transform(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError("Expected a finite proper rigid transform")
    return matrix


def _stage(path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None or UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Bundled visual assets must use metres and Z up")
    return stage


def _material(prim, asset_root):
    from pxr import UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shader = material.ComputeSurfaceSource()[0] if material else None
    color, roughness, metallic, opacity, texture = (0.7, 0.7, 0.7), 0.5, 0.0, 1.0, None
    if shader:
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            raise ValueError(f"Unsupported material shader on {prim.GetPath()}")
        for name, default in (
            ("diffuseColor", color),
            ("roughness", roughness),
            ("metallic", metallic),
            ("opacity", opacity),
        ):
            inp = shader.GetInput(name)
            value = inp.Get() if inp else None
            value = default if value is None else value
            if name == "diffuseColor":
                color = tuple(float(c) for c in value)
                source = inp.GetConnectedSource() if inp else None
                if source:
                    texture_shader = UsdShade.Shader(source[0].GetPrim())
                    if texture_shader.GetIdAttr().Get() != "UsdUVTexture":
                        raise ValueError("Only direct USD UV diffuse textures are supported")
                    file = texture_shader.GetInput("file").Get()
                    texture = Path(file.resolvedPath).resolve()
                    if not texture.is_relative_to(asset_root) or not texture.is_file():
                        raise ValueError("Texture must resolve inside the bundled asset tree")
            elif name == "roughness":
                roughness = float(value)
            elif name == "metallic":
                metallic = float(value)
            else:
                opacity = float(value)
    values = np.asarray((*color, roughness, metallic, opacity))
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Material parameters must be finite values in [0,1]")
    return color, roughness, metallic, opacity, texture


def _mesh(prim, transform, *, name, frame, asset_root):
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Visual mesh positions must be finite [N,3]")
    if not len(counts) or np.any(counts != 3) or len(indices) != 3 * len(counts):
        raise ValueError("Bundled visual meshes must already be triangulated")
    if np.any(indices < 0) or np.any(indices >= len(vertices)):
        raise ValueError("Visual mesh triangle indices are out of bounds")
    triangles = indices.reshape(-1, 3)
    # Preserve seams if an asset supplies face-varying UVs. The original-point
    # map lets a deformable renderer update every duplicated corner correctly.
    vertex_indices = np.arange(len(vertices), dtype=np.int32)
    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    uvs = None
    if uv and uv.HasValue():
        uvs = np.asarray(uv.ComputeFlattened(), dtype=np.float32)
        if uv.GetInterpolation() == "faceVarying":
            if uvs.shape != (len(indices), 2):
                raise ValueError("Face-varying UVs must match triangle corners")
            vertex_indices = indices.copy()
            vertices = vertices[vertex_indices]
            triangles = np.arange(len(indices), dtype=np.int32).reshape(-1, 3)
        elif uv.GetInterpolation() not in ("vertex", "varying") or uvs.shape != (len(vertices), 2):
            raise ValueError("Expected vertex or face-varying texture coordinates")
        if not np.isfinite(uvs).all():
            raise ValueError("Texture coordinates must be finite")
    transform = _transform(transform)
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        triangles = triangles[:, [0, 2, 1]].copy()
    color, roughness, metallic, opacity, texture = _material(prim, asset_root)
    if texture is not None and uvs is None:
        raise ValueError("A textured mesh must have authored UV coordinates")
    return VisualMesh(
        name,
        vertices.astype(np.float32),
        triangles.copy(),
        vertex_indices,
        color,
        roughness,
        metallic,
        opacity,
        uvs,
        texture,
        str(prim.GetPath()),
        frame,
    )


def _provenance(stage, root):
    result = []
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        path = Path(layer.realPath).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Visual asset references leave the bundled tree")
        result.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sorted(result, key=lambda entry: entry["path"])


def load_native_phantom(recording=None) -> VisualAsset:
    """Load all authored phantom surfaces aligned to the native FEM pad frame.

    No inactive-region cut faces are introduced. Vertices are original regional
    surface vertices; updating the active region is the renderer's responsibility.
    """
    from pxr import UsdGeom

    path, manifest, root, scene = _recording(recording)
    stage = _stage(scene)
    native_from_world = np.linalg.inv(_transform(manifest["phantom"]["T_world_from_native"]))
    cache = UsdGeom.XformCache()
    meshes = []
    for region in sorted(manifest["phantom"]["regions"], key=lambda value: value["id"]):
        name = f"Region{region['id']:02d}"
        prim = stage.GetPrimAtPath(f"/SutureBotReplay/Phantom/{name}")
        if not prim or prim.GetTypeName() != "Mesh":
            raise ValueError(f"Missing phantom surface {name}")
        world_from_local = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        meshes.append(
            _mesh(prim, native_from_world @ world_from_local, name=name, frame="native FEM pad", asset_root=root)
        )
    return VisualAsset(
        tuple(meshes),
        {
            "kind": "full authored phantom",
            "sample_id": manifest["sample_id"],
            "recording": str(path.relative_to(root)),
            "native_from_recording_world": native_from_world.tolist(),
            "appearance": "original USD materials; no photographic tissue texture is bundled",
            "calibration": "recorded placement is a fitted estimate; inverse placement restores native simulation coordinates",
            "source_layers": _provenance(stage, root),
        },
        {},
    )


def _joint_frame(joint, end):
    from pxr import Gf

    position = joint.GetAttribute(f"physics:localPos{end}").Get()
    rotation = joint.GetAttribute(f"physics:localRot{end}").Get()
    result = np.eye(4)
    result[:3, :3] = np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(rotation)))).T
    result[:3, 3] = np.asarray(position)
    return _transform(result)


def load_dvrk_tool(recording=None, *, instrument="PSM1", jaw_opening=0.0, shaft_length_m=None) -> VisualAsset:
    """Load only a distal tool, in its wrist-yaw frame at zero pitch/yaw.

    ``jaw_opening`` is total symmetric jaw angle [rad]. The five original CAD
    components retain their material appearance. Optional ``shaft_length_m``
    matches the physical shaft capsule's centreline length by moving only the
    proximal tube end. Its diameter, distal detail, wrists and jaws retain their
    source dimensions. Omit it for the full source CAD. This render-only assembly
    creates no robot arms, recorded motion, collisions or grasp constraints.
    """
    from pxr import UsdGeom

    if instrument not in ("PSM1", "PSM2") or not np.isfinite(jaw_opening) or jaw_opening < 0:
        raise ValueError("Expected PSM1 or PSM2 and a finite nonnegative jaw opening")
    source_shaft_length = 0.547
    shaft_length = source_shaft_length if shaft_length_m is None else float(shaft_length_m)
    if not np.isfinite(shaft_length) or not 0 < shaft_length <= source_shaft_length:
        raise ValueError("Shaft centreline length must be finite and in (0, 0.547] metres")
    path, _manifest, root, scene = _recording(recording)
    stage = _stage(scene)
    rig = _stage(root / "dvrk/suturebot_rig.usda")
    roll_from_body = {"roll": np.eye(4)}
    for name, parent, child, angle in (
        ("wrist_pitch", "roll", "pitch", 0.0),
        ("wrist_yaw", "pitch", "yaw", 0.0),
        ("jaw_left", "yaw", "jaw1", jaw_opening / 2),
        ("jaw_right", "yaw", "jaw2", -jaw_opening / 2),
    ):
        joint = rig.GetPrimAtPath(f"/SutureBot/{instrument}/Joints/{name}")
        if not joint:
            raise ValueError(f"Missing source joint {name}")
        lower, upper = (np.deg2rad(joint.GetAttribute(f"physics:{side}Limit").Get()) for side in ("lower", "upper"))
        if not lower - 1e-8 <= angle <= upper + 1e-8:
            raise ValueError("Jaw opening exceeds the source model joint limits")
        if joint.GetAttribute("physics:axis").Get() != "Z":
            raise ValueError("Unexpected source tool joint axis")
        c, s = np.cos(angle), np.sin(angle)
        motion = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        roll_from_body[child] = (
            roll_from_body[parent] @ _joint_frame(joint, 0) @ motion @ np.linalg.inv(_joint_frame(joint, 1))
        )
    tool_from_roll = np.linalg.inv(roll_from_body["yaw"])
    transforms = {name: tool_from_roll @ pose for name, pose in roll_from_body.items()}
    cache = UsdGeom.XformCache()
    meshes = []
    for name, transform in transforms.items():
        body = stage.GetPrimAtPath(f"/SutureBotReplay/{instrument}/{name}")
        prim = stage.GetPrimAtPath(f"/SutureBotReplay/{instrument}/{name}/CAD")
        if not body or not prim:
            raise ValueError(f"Missing source tool component {name}")
        body_from_mesh = (
            np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)
            @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        )
        mesh = _mesh(prim, transform @ body_from_mesh, name=name, frame=f"{instrument} wrist-yaw", asset_root=root)
        if name == "roll" and shaft_length < source_shaft_length:
            vertices = mesh.vertices.copy()
            # The source tube connects its proximal end to the distal ring at
            # y=-18.332 mm. Move the rear end only; do not scale the wrist detail.
            vertices[vertices[:, 1] < -0.02, 1] += source_shaft_length - shaft_length
            mesh = replace(mesh, vertices=vertices)
        meshes.append(mesh)
    return VisualAsset(
        tuple(meshes),
        {
            "kind": "distal tool CAD with a configurable shaft; visual assembly only",
            "instrument": instrument,
            "recording": str(path.relative_to(root)),
            "coordinate_frame": f"{instrument} wrist-yaw body",
            "wrist_pitch_rad": 0.0,
            "wrist_yaw_rad": 0.0,
            "symmetric_jaw_opening_rad": float(jaw_opening),
            "shaft_centerline_length_m": shaft_length,
            "shaft_geometry": "source diameter and distal detail; proximal tube end follows physical shaft length",
            "appearance": "composed source USD materials; metallic layers are illustrative, not calibrated optics",
            "physics": "no articulation, grasp or contact is created by this loader",
            "source_layers": _provenance(stage, root),
            "joint_source_layers": _provenance(rig, root),
        },
        transforms,
    )


def load_needle_visual(path=None) -> VisualAsset:
    """Original tapered needle CAD in its independent rigid-body frame.

    The bundled material is a gray diffuse placeholder. This view explicitly
    assigns an illustrative polished steel material, without changing geometry
    or the capsule collision model.
    """
    from pxr import UsdGeom

    from newton_solver_suturebot import NeedleGeometry

    path = Path(NeedleGeometry.load().asset_path if path is None else path).resolve()
    stage = _stage(path)
    body = stage.GetPrimAtPath("/Needle")
    prim = stage.GetPrimAtPath("/Needle/OriginalCAD")
    if not body or not prim:
        raise ValueError("Needle asset must contain its original CAD mesh")
    cache = UsdGeom.XformCache()
    transform = (
        np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)
        @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
    )
    mesh = _mesh(prim, transform, name="needle", frame="Needle rigid body", asset_root=path.parent)
    authored = UsdGeom.Mesh(prim).GetNormalsAttr().Get()
    if authored is not None and UsdGeom.Mesh(prim).GetNormalsInterpolation() == "faceVarying":
        # The asset carries one normal per triangle corner, smooth along the wire and sharp at
        # its real edges. They belong to the corners that face-varying UVs make into vertices.
        normals = np.asarray(authored, dtype=np.float64) @ transform[:3, :3].T
        if normals.shape != mesh.vertices.shape:
            raise ValueError("Face-varying needle normals need face-varying texture coordinates")
    else:
        vertices, triangles = mesh.vertices, mesh.triangles
        # Area-weighted smooth normals retain the source geometry and taper.
        normals = np.zeros_like(vertices)
        faces = vertices[triangles]
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        for corner in range(3):
            np.add.at(normals, triangles[:, corner], cross)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-15):
        raise ValueError("Needle CAD contains a vertex without a valid surface normal")
    normals = (normals / lengths[:, None]).astype(np.float32)
    mesh = replace(mesh, diffuse_rgb=(0.68, 0.70, 0.73), metallic=1.0, roughness=0.19, normals=normals)
    return VisualAsset(
        (mesh,),
        {
            "kind": "original tapered needle CAD",
            "coordinate_frame": "independent Needle rigid body",
            "material": "illustrative polished steel, metallic=1, roughness=0.19; optical properties uncalibrated",
            "source_layers": _provenance(stage, path.parent),
        },
        {"needle": transform},
    )
