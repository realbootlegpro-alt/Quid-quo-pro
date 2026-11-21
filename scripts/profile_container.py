#!/usr/bin/env python3
"""Pre-Production Asset Inspector & Normalizer (scripts/profile_container.py).

Outer Python entrypoint:
    python scripts/profile_container.py [--force] [--blender /path/to/blender] [--container-dir import_container]

Dual-mode execution:
  1. Outer Python:
     - Scans import_container/characters/ and import_container/props/ for .glb/.blend.
     - Compares file byte size & mtime_ns against _index/file_hashes.json.
     - Prunes deleted source files and orphaned sidecars cleanly.
     - Launches Blender 4.5 LTS in background (--factory-startup) for new/changed assets.
     - Uses Pillow to draw labeled 0.1m X/Y/Z ruler grids onto character orthographic
       views, stitches 2x2 *_grid.png (characters) or 1x2 *_thumb.png (props), and
       writes _index/asset_catalog.json and _index/file_hashes.json.
  2. Headless Blender Worker (--blender-worker):
     - Resets to empty factory scene (read_factory_settings(use_empty=True)).
     - Imports .glb or appends .blend cleanly, strips embedded cameras/lights/probes,
       wraps root objects in a fresh Empty at (0, 0, 0) preserving glTF +90° X orientation
       and armature rotations, and grounds the lowest evaluated mesh vertex at z = 0.0.
     - Normalizes unlit / KHR_materials_unlit / Emission-only game-rip shaders into
       Blender 4.5 Principled BSDF materials.
     - Runs category-specific bmesh / armature / ray_cast inspection.
     - Packs textures and saves <id>_norm.blend, then renders raw views via
       BLENDER_EEVEE_NEXT and exports exact orthographic meter-to-pixel bounds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".glb", ".blend"}
CATEGORIES = ("characters", "props")
CONTAINER_KEYWORDS = {"bowl", "pot", "cup", "ramekin", "pan"}
CONTAINER_TAGS = CONTAINER_KEYWORDS | {
    "container",
    "vessel",
    "skillet",
    "saucepan",
    "stockpot",
    "mug",
}

PANEL_SIZE_PX = 768
CHARACTER_HEADER_PX = 56
PROP_HEADER_PX = 68

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------


def sanitize_slug(stem: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return slug or "asset"


def tokenize_name(name: str) -> list[str]:
    """Split snake_case, kebab-case, and CamelCase into lowercase word tokens."""
    parts = re.findall(r"[A-Za-z0-9]+", name)
    tokens: list[str] = []
    for part in parts:
        sub = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", part)
        tokens.extend(t.lower() for t in (sub or [part]))
    return tokens


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp_path, path)


def round_vec(vec: Any, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in vec]


# ===========================================================================
# HEADLESS BLENDER WORKER (bpy / bmesh / mathutils)
# ===========================================================================


def blender_worker_main(argv: list[str]) -> int:
    import bpy  # type: ignore
    import bmesh  # type: ignore
    import numpy as np  # type: ignore
    from addon_utils import check as addon_check, enable as addon_enable  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore

    parser = argparse.ArgumentParser(description="Blender worker for profile_container.py")
    parser.add_argument("--src", required=True, help="Source .glb or .blend path")
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--norm-blend", required=True, help="Output normalized .blend path")
    parser.add_argument("--raw-dir", required=True, help="Temp folder for raw PNG views")
    parser.add_argument("--worker-json", required=True, help="Output worker JSON path")
    args = parser.parse_args(argv)

    src_path = Path(args.src).resolve()
    norm_blend_path = Path(args.norm_blend).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    worker_json_path = Path(args.worker_json).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_blend_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Clean Empty Factory Scene & Import / Append
    # ------------------------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    # Ensure built-in glTF 2.0 importer stays enabled after factory reset
    _, gltf_enabled = addon_check("io_scene_gltf2")
    if not gltf_enabled or not hasattr(bpy.ops.import_scene, "gltf"):
        addon_enable("io_scene_gltf2", default_set=False, persistent=False)

    ext = src_path.suffix.lower()
    bone_shape_objs = set()

    if ext == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(src_path))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(src_path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
            data_to.actions = list(data_from.actions)

        for obj in data_to.objects:
            if obj is not None and obj.name not in scene.collection.objects:
                scene.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported asset extension: {ext}")

    # Strip stray cameras/lights and hide custom bone shapes for BOTH .glb and .blend
    for obj in list(scene.objects):
        if obj.type == "ARMATURE" and obj.pose:
            for pbone in obj.pose.bones:
                if pbone.custom_shape is not None:
                    bone_shape_objs.add(pbone.custom_shape)

    for obj in list(scene.objects):
        if obj.type in {"CAMERA", "LIGHT", "SPEAKER", "LIGHT_PROBE"}:
            bpy.data.objects.remove(obj, do_unlink=True)
        elif obj in bone_shape_objs:
            obj.hide_render = True
            obj.hide_viewport = True

    bpy.context.view_layer.update()

    # Identify active asset objects and renderable geometry objects
    asset_objects = list(scene.objects)
    if not asset_objects:
        raise RuntimeError(f"No objects found after importing {src_path.name}")

    def is_renderable_geo(obj: Any) -> bool:
        if obj.hide_render:
            return False
        if ext == ".blend" and obj.hide_viewport:
            return False
        return obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}

    geo_objects = [o for o in asset_objects if is_renderable_geo(o)]
    if not geo_objects:
        raise RuntimeError(f"No visible mesh/geometry objects found in {src_path.name}")

    # Wrap all current root objects inside a fresh Empty at (0, 0, 0)
    # Using matrix_parent_inverse = Identity preserves glTF +90° X rotation
    # and all local/armature transforms untouched.
    root_objects = [o for o in asset_objects if o.parent is None]
    asset_root = bpy.data.objects.new(f"{args.asset_id}_root", None)
    asset_root.empty_display_type = "PLAIN_AXES"
    asset_root.empty_display_size = 0.25
    asset_root.location = (0.0, 0.0, 0.0)
    asset_root.rotation_euler = (0.0, 0.0, 0.0)
    asset_root.scale = (1.0, 1.0, 1.0)
    scene.collection.objects.link(asset_root)

    for root_obj in root_objects:
        root_obj.parent = asset_root
        root_obj.matrix_parent_inverse = Matrix.Identity(4)

    bpy.context.view_layer.update()

    def compute_world_bounds(objs: list[Any]) -> dict[str, Any]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        total_verts = 0

        for obj in objs:
            obj_eval = obj.evaluated_get(depsgraph)
            mesh_eval = None
            try:
                mesh_eval = obj_eval.to_mesh()
                if mesh_eval is None or len(mesh_eval.vertices) == 0:
                    continue
                n_verts = len(mesh_eval.vertices)
                coords = np.empty(n_verts * 3, dtype=np.float64)
                mesh_eval.vertices.foreach_get("co", coords)
                coords = coords.reshape((n_verts, 3))

                mat = np.asarray(obj_eval.matrix_world, dtype=np.float64)
                world_coords = coords @ mat[:3, :3].T + mat[:3, 3]

                mins = np.minimum(mins, world_coords.min(axis=0))
                maxs = np.maximum(maxs, world_coords.max(axis=0))
                total_verts += n_verts
            finally:
                if mesh_eval is not None:
                    obj_eval.to_mesh_clear()

        if total_verts == 0 or not np.all(np.isfinite(mins)):
            raise RuntimeError(f"All geometry objects in {src_path.name} have 0 vertices")

        dims = maxs - mins
        center = 0.5 * (mins + maxs)
        return {
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "center": center.tolist(),
            "dimensions_m": dims.tolist(),
            "total_evaluated_vertices": int(total_verts),
        }

    pre_bounds = compute_world_bounds(geo_objects)
    pre_ground_min_z = float(pre_bounds["min"][2])

    # Shift parent Empty along Z so lowest vertex sits at z = 0.0
    asset_root.location.z = -pre_ground_min_z
    bpy.context.view_layer.update()

    world_bounds = compute_world_bounds(geo_objects)
    # Clamp numerical epsilon around 0.0
    if abs(world_bounds["min"][2]) < 1e-6:
        world_bounds["min"][2] = 0.0

    # ------------------------------------------------------------------
    # 2. Shader Normalization (Blender 4.5 Principled BSDF)
    # ------------------------------------------------------------------
    def find_socket(node: Any, *names: str) -> Any:
        for name in names:
            if name in node.inputs:
                return node.inputs[name]
        return None

    def to_rgba(val: Any) -> tuple[float, float, float, float]:
        seq = list(val)
        if len(seq) == 3:
            return (float(seq[0]), float(seq[1]), float(seq[2]), 1.0)
        if len(seq) >= 4:
            return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
        return (0.8, 0.8, 0.8, 1.0)

    def collect_upstream(socket: Any, visited: set[Any] | None = None) -> set[Any]:
        if visited is None:
            visited = set()
        if socket is None or not socket.is_linked:
            return visited
        for link in socket.links:
            src_node = link.from_node
            if src_node not in visited:
                visited.add(src_node)
                for inp in src_node.inputs:
                    collect_upstream(inp, visited)
        return visited

    def upstream_image_keys(socket: Any) -> set[str]:
        keys: set[str] = set()
        for node in collect_upstream(socket):
            if node.type == "TEX_IMAGE":
                if node.image is not None:
                    keys.add(node.image.filepath_raw or node.image.name)
                else:
                    keys.add(node.name)
        return keys

    def is_light_path_node(socket: Any) -> bool:
        if not socket.is_linked:
            return False
        return any(n.type == "LIGHT_PATH" for n in collect_upstream(socket))

    def unwrap_unlit_chain(
        shader_socket: Any, depth: int = 0
    ) -> tuple[Any, Any, Any, bool]:
        """Return (color_socket, color_default, alpha_socket, alpha_inverted)."""
        if shader_socket is None or not shader_socket.is_linked or depth > 12:
            return (None, None, None, False)

        src_node = shader_socket.links[0].from_node
        ntype = src_node.type

        if ntype in {"EMISSION", "BACKGROUND", "BSDF_DIFFUSE"}:
            col_in = find_socket(src_node, "Color")
            if col_in is not None:
                if col_in.is_linked:
                    return (col_in.links[0].from_socket, None, None, False)
                return (None, to_rgba(col_in.default_value), None, False)

        if ntype == "MIX_SHADER":
            fac_in = src_node.inputs[0]
            sh1_in = src_node.inputs[1]
            sh2_in = src_node.inputs[2]
            sh1_node = sh1_in.links[0].from_node if sh1_in.is_linked else None
            sh2_node = sh2_in.links[0].from_node if sh2_in.is_linked else None

            # glTF KHR_materials_unlit uses Light Path (Is Camera Ray) -> Mix(Transparent, Unlit)
            if is_light_path_node(fac_in):
                if sh1_node and sh1_node.type == "BSDF_TRANSPARENT":
                    return unwrap_unlit_chain(sh2_in, depth + 1)
                if sh2_node and sh2_node.type == "BSDF_TRANSPARENT":
                    return unwrap_unlit_chain(sh1_in, depth + 1)

            # Real cutout/alpha mix with Transparent BSDF
            if sh1_node and sh1_node.type == "BSDF_TRANSPARENT":
                c_sock, c_def, _, _ = unwrap_unlit_chain(sh2_in, depth + 1)
                a_sock = fac_in.links[0].from_socket if fac_in.is_linked else None
                return (c_sock, c_def, a_sock, False)
            if sh2_node and sh2_node.type == "BSDF_TRANSPARENT":
                c_sock, c_def, _, _ = unwrap_unlit_chain(sh1_in, depth + 1)
                a_sock = fac_in.links[0].from_socket if fac_in.is_linked else None
                return (c_sock, c_def, a_sock, True)

            # Generic Mix Shader fallback: prefer non-transparent shader
            c_sock, c_def, a_sock, a_inv = unwrap_unlit_chain(sh2_in, depth + 1)
            if c_sock is not None or c_def is not None:
                return (c_sock, c_def, a_sock, a_inv)
            return unwrap_unlit_chain(sh1_in, depth + 1)

        if ntype in {
            "TEX_IMAGE",
            "RGB",
            "ATTRIBUTE",
            "VERTEX_COLOR",
            "MIX",
            "MIX_RGB",
            "HUE_SAT",
            "BRIGHTCONTRAST",
            "GAMMA",
            "VALTORGB",
        }:
            return (shader_socket.links[0].from_socket, None, None, False)

        return (None, None, None, False)

    def normalize_material(mat: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "material": mat.name,
            "action": "unchanged",
            "reason": None,
        }

        if getattr(mat, "is_grease_pencil", False):
            return info

        if not mat.use_nodes or mat.node_tree is None:
            base_rgba = to_rgba(getattr(mat, "diffuse_color", (0.8, 0.8, 0.8, 1.0)))
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            out_node = nt.nodes.new("ShaderNodeOutputMaterial")
            out_node.location = (300, 0)
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            find_socket(bsdf, "Base Color").default_value = base_rgba
            find_socket(bsdf, "Roughness").default_value = 0.65
            spec_sock = find_socket(bsdf, "Specular IOR Level", "Specular")
            if spec_sock:
                spec_sock.default_value = 0.35
            emit_str = find_socket(bsdf, "Emission Strength")
            if emit_str:
                emit_str.default_value = 0.0
            nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
            info["action"] = "converted_non_node"
            info["reason"] = "non_node_material"
            return info

        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links

        out_node = next(
            (n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
            None,
        ) or next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)

        if out_node is None:
            out_node = nodes.new("ShaderNodeOutputMaterial")
            out_node.location = (400, 0)

        surf_in = out_node.inputs.get("Surface")
        if surf_in is None or not surf_in.is_linked:
            return info

        upstream = collect_upstream(surf_in)
        principled = next((n for n in upstream if n.type == "BSDF_PRINCIPLED"), None)

        # Case 1: Unlit / Emission / Background / Direct-Texture tree (no Principled BSDF)
        if principled is None:
            unlit_types = {"EMISSION", "BACKGROUND", "LIGHT_PATH"}
            direct_color_types = {
                "TEX_IMAGE",
                "RGB",
                "ATTRIBUTE",
                "VERTEX_COLOR",
                "MIX",
                "MIX_RGB",
                "VALTORGB",
            }
            root_shader = surf_in.links[0].from_node
            is_unlit_setup = bool(
                any(n.type in unlit_types for n in upstream)
                or root_shader.type in direct_color_types
                or root_shader.type == "BSDF_DIFFUSE"
            )
            if not is_unlit_setup:
                return info

            c_sock, c_def, a_sock, a_inv = unwrap_unlit_chain(surf_in)
            if c_sock is None and c_def is None:
                tex = next((n for n in upstream if n.type == "TEX_IMAGE"), None)
                if tex is not None:
                    c_sock = tex.outputs["Color"]

            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (out_node.location.x - 300, out_node.location.y)

            base_in = find_socket(bsdf, "Base Color")
            rough_in = find_socket(bsdf, "Roughness")
            spec_in = find_socket(bsdf, "Specular IOR Level", "Specular")
            emit_col_in = find_socket(bsdf, "Emission Color", "Emission")
            emit_str_in = find_socket(bsdf, "Emission Strength")
            alpha_in = find_socket(bsdf, "Alpha")

            if c_sock is not None:
                links.new(c_sock, base_in)
            elif c_def is not None:
                base_in.default_value = c_def

            if a_sock is not None and alpha_in is not None:
                if a_inv:
                    inv_node = nodes.new("ShaderNodeMath")
                    inv_node.operation = "SUBTRACT"
                    inv_node.location = (bsdf.location.x - 180, bsdf.location.y - 220)
                    inv_node.inputs[0].default_value = 1.0
                    links.new(a_sock, inv_node.inputs[1])
                    links.new(inv_node.outputs["Value"], alpha_in)
                else:
                    links.new(a_sock, alpha_in)

            if rough_in is not None:
                rough_in.default_value = 0.65
            if spec_in is not None:
                spec_in.default_value = 0.35
            if emit_col_in is not None:
                emit_col_in.default_value = (0.0, 0.0, 0.0, 1.0)
            if emit_str_in is not None:
                emit_str_in.default_value = 0.0

            # Replace surface link
            for l in list(surf_in.links):
                links.remove(l)
            links.new(bsdf.outputs["BSDF"], surf_in)

            # Prune disconnected legacy unlit shader nodes
            keep_upstream = collect_upstream(surf_in) | {out_node}
            for node in list(nodes):
                if node not in keep_upstream and node.type in {
                    "EMISSION",
                    "BACKGROUND",
                    "MIX_SHADER",
                    "LIGHT_PATH",
                    "BSDF_TRANSPARENT",
                    "BSDF_DIFFUSE",
                }:
                    nodes.remove(node)

            info["action"] = "rewired_to_principled"
            info["reason"] = "unlit_or_emission_graph"
            return info

        # Case 2: Principled BSDF exists -> inspect for Emission-only / over-lit game rips
        base_in = find_socket(principled, "Base Color")
        rough_in = find_socket(principled, "Roughness")
        spec_in = find_socket(principled, "Specular IOR Level", "Specular")
        emit_col_in = find_socket(principled, "Emission Color", "Emission")
        emit_str_in = find_socket(principled, "Emission Strength")

        if base_in is None or emit_col_in is None:
            return info

        emit_linked = emit_col_in.is_linked
        emit_col_val = to_rgba(emit_col_in.default_value)
        emit_has_color = emit_linked or max(emit_col_val[:3]) > 0.01

        emit_str_linked = bool(emit_str_in and emit_str_in.is_linked)
        emit_str_val = (
            float(emit_str_in.default_value)
            if (emit_str_in and not emit_str_linked)
            else (1.0 if emit_str_linked else 0.0)
        )
        # Note: in glTF, emissionColor without explicit strength defaults strength=1.0
        emission_active = emit_has_color and (emit_str_val > 0.0 or emit_str_linked)

        if not emission_active:
            return info

        base_default = to_rgba(base_in.default_value)
        base_is_blank = (not base_in.is_linked) and (
            max(base_default[:3]) <= 0.02
            or all(abs(c - 0.8) < 0.01 for c in base_default[:3])
        )

        # Subcase 2A: Emission-only (Base Color blank/black, texture/color in Emission)
        if base_is_blank:
            if emit_linked:
                src_sock = emit_col_in.links[0].from_socket
                for l in list(emit_col_in.links):
                    links.remove(l)
                links.new(src_sock, base_in)
            else:
                base_in.default_value = emit_col_val

            emit_col_in.default_value = (0.0, 0.0, 0.0, 1.0)
            if emit_str_in is not None:
                for l in list(emit_str_in.links):
                    links.remove(l)
                emit_str_in.default_value = 0.0

            if rough_in is not None and not rough_in.is_linked:
                if rough_in.default_value >= 0.99 or rough_in.default_value <= 0.01:
                    rough_in.default_value = 0.65
            if spec_in is not None and not spec_in.is_linked:
                if spec_in.default_value <= 0.01:
                    spec_in.default_value = 0.35

            info["action"] = "rewired_emission_to_base_color"
            info["reason"] = "emission_only_principled"
            return info

        # Subcase 2B: Base Color exists AND Emission duplicates Base texture or washes it out uniformly
        if base_in.is_linked:
            base_imgs = upstream_image_keys(base_in)
            emit_imgs = upstream_image_keys(emit_col_in) if emit_linked else set()
            shares_albedo = bool(base_imgs and emit_imgs and (base_imgs & emit_imgs))
            uniform_washout = (not emit_linked) and (not emit_str_linked) and emit_str_val > 0.0

            if shares_albedo or uniform_washout:
                for l in list(emit_col_in.links):
                    links.remove(l)
                emit_col_in.default_value = (0.0, 0.0, 0.0, 1.0)
                if emit_str_in is not None:
                    for l in list(emit_str_in.links):
                        links.remove(l)
                    emit_str_in.default_value = 0.0
                if rough_in is not None and not rough_in.is_linked:
                    if rough_in.default_value >= 0.99:
                        rough_in.default_value = 0.65
                if spec_in is not None and not spec_in.is_linked:
                    if spec_in.default_value <= 0.01:
                        spec_in.default_value = 0.35

                info["action"] = "zeroed_unwanted_emission"
                info["reason"] = (
                    "shared_albedo_emission" if shares_albedo else "uniform_flat_emission"
                )
                return info

        return info

    shader_normalization = [normalize_material(m) for m in bpy.data.materials]

    # ------------------------------------------------------------------
    # 3. Category-Specific Inspection
    # ------------------------------------------------------------------
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": args.asset_id,
        "category": args.category,
        "source_file": src_path.name,
        "source_ext": ext,
        "blender_version": bpy.app.version_string,
        "dimensions_m": round_vec(world_bounds["dimensions_m"]),
        "bounds_m": {
            "min": round_vec(world_bounds["min"]),
            "max": round_vec(world_bounds["max"]),
            "center": round_vec(world_bounds["center"]),
        },
        "total_evaluated_vertices": world_bounds["total_evaluated_vertices"],
        "shader_normalization": shader_normalization,
    }

    if args.category == "characters":
        # 3A. Character inspection: bmesh connectivity, slots, bones, animations
        total_loose_islands = 0
        material_slots_meta: list[dict[str, Any]] = []

        for obj in geo_objects:
            obj_eval = obj.evaluated_get(depsgraph)
            mesh_eval = None
            bm = bmesh.new()
            try:
                mesh_eval = obj_eval.to_mesh()
                if mesh_eval is None or len(mesh_eval.vertices) == 0:
                    continue

                bm.from_mesh(mesh_eval)
                # Weld coincident UV/normal-split vertices strictly on the temporary
                # inspection bmesh so UV seams do not inflate loose_island_count or
                # falsely mark closed UV-mapped shells as open!
                bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()
                bm.verts.index_update()
                bm.edges.index_update()
                bm.faces.index_update()

                # Count loose islands via non-destructive bmesh traversal
                visited_verts: set[int] = set()
                has_faces = len(bm.faces) > 0
                obj_islands = 0
                for v in bm.verts:
                    if v.index in visited_verts:
                        continue
                    if has_faces and not v.link_faces and not v.link_edges:
                        visited_verts.add(v.index)
                        continue
                    obj_islands += 1
                    queue = deque([v])
                    visited_verts.add(v.index)
                    while queue:
                        curr = queue.popleft()
                        for e in curr.link_edges:
                            other = e.other_vert(curr)
                            if other.index not in visited_verts:
                                visited_verts.add(other.index)
                                queue.append(other)

                total_loose_islands += obj_islands

                # Inspect each material slot
                slot_names: list[str] = []
                if len(obj_eval.material_slots) > 0:
                    for slot in obj_eval.material_slots:
                        slot_names.append(slot.material.name if slot.material else "<empty>")
                else:
                    slot_names = ["<default>"]

                faces_by_slot: dict[int, list[Any]] = {i: [] for i in range(len(slot_names))}
                for f in bm.faces:
                    idx = f.material_index if 0 <= f.material_index < len(slot_names) else 0
                    faces_by_slot.setdefault(idx, []).append(f)

                mat_world = obj_eval.matrix_world
                for slot_idx, mat_name in enumerate(slot_names):
                    slot_faces = faces_by_slot.get(slot_idx, [])
                    if not slot_faces:
                        material_slots_meta.append(
                            {
                                "object": obj.name,
                                "slot_index": slot_idx,
                                "material": mat_name,
                                "face_count": 0,
                                "vertex_count": 0,
                                "center_m": None,
                                "size_m": None,
                                "is_closed_shell": False,
                                "open_boundary_edges": 0,
                            }
                        )
                        continue

                    slot_verts_set = set()
                    edge_face_counts: dict[int, int] = {}
                    for f in slot_faces:
                        for v in f.verts:
                            slot_verts_set.add(v)
                        for e in f.edges:
                            edge_face_counts[e.index] = edge_face_counts.get(e.index, 0) + 1

                    open_boundary_edges = sum(
                        1 for cnt in edge_face_counts.values() if cnt == 1
                    )
                    is_closed_shell = open_boundary_edges == 0

                    world_pts = [mat_world @ v.co for v in slot_verts_set]
                    s_min = [
                        min(p.x for p in world_pts),
                        min(p.y for p in world_pts),
                        min(p.z for p in world_pts),
                    ]
                    s_max = [
                        max(p.x for p in world_pts),
                        max(p.y for p in world_pts),
                        max(p.z for p in world_pts),
                    ]
                    s_center = [0.5 * (mn + mx) for mn, mx in zip(s_min, s_max)]
                    s_size = [mx - mn for mn, mx in zip(s_min, s_max)]

                    material_slots_meta.append(
                        {
                            "object": obj.name,
                            "slot_index": slot_idx,
                            "material": mat_name,
                            "face_count": len(slot_faces),
                            "vertex_count": len(slot_verts_set),
                            "center_m": round_vec(s_center),
                            "size_m": round_vec(s_size),
                            "is_closed_shell": bool(is_closed_shell),
                            "open_boundary_edges": int(open_boundary_edges),
                        }
                    )
            finally:
                bm.free()
                if mesh_eval is not None:
                    obj_eval.to_mesh_clear()

        # Armatures & bones
        armature_objs = [o for o in asset_objects if o.type == "ARMATURE"]
        bones_flat: list[str] = []
        armature_details: list[dict[str, Any]] = []
        for arm in armature_objs:
            arm_bones: list[dict[str, Any]] = []
            for bone in arm.data.bones:
                bones_flat.append(bone.name)
                head_w = arm.matrix_world @ bone.head_local
                tail_w = arm.matrix_world @ bone.tail_local
                arm_bones.append(
                    {
                        "name": bone.name,
                        "parent": bone.parent.name if bone.parent else None,
                        "head_m": round_vec(head_w),
                        "tail_m": round_vec(tail_w),
                        "length_m": round(float((tail_w - head_w).length), 6),
                    }
                )
            armature_details.append(
                {
                    "name": arm.name,
                    "bone_count": len(arm_bones),
                    "bones": arm_bones,
                }
            )

        # Embedded animations (actions & NLA tracks)
        fps = float(scene.render.fps / scene.render.fps_base) if scene.render.fps_base else 24.0
        seen_actions: dict[str, Any] = {}
        for act in bpy.data.actions:
            seen_actions[act.name] = act
        for obj in asset_objects:
            ad = getattr(obj, "animation_data", None)
            if not ad:
                continue
            if ad.action is not None:
                seen_actions[ad.action.name] = ad.action
            for track in ad.nla_tracks:
                for strip in track.strips:
                    if strip.action is not None:
                        seen_actions[strip.action.name] = strip.action

        embedded_animations = sorted(seen_actions.keys())
        animation_clips: list[dict[str, Any]] = []
        for name in embedded_animations:
            act = seen_actions[name]
            f_start, f_end = act.frame_range
            duration_s = max(0.0, float(f_end - f_start) / fps) if fps > 0 else 0.0
            animation_clips.append(
                {
                    "name": name,
                    "frame_start": round(float(f_start), 3),
                    "frame_end": round(float(f_end), 3),
                    "duration_s": round(duration_s, 4),
                    "fps": round(fps, 3),
                }
            )

        meta["loose_island_count"] = int(total_loose_islands)
        meta["bones"] = bones_flat
        meta["armatures"] = armature_details
        meta["embedded_animations"] = embedded_animations
        meta["animation_clips"] = animation_clips
        meta["material_slots"] = material_slots_meta

    else:
        # 3B. Prop inspection: bottom_z_offset, materials[], container ray_cast cavity_bounds
        prop_materials: list[str] = []
        seen_mat_names: set[str] = set()
        for obj in geo_objects:
            for slot in obj.material_slots:
                if slot.material and slot.material.name not in seen_mat_names:
                    seen_mat_names.add(slot.material.name)
                    prop_materials.append(slot.material.name)

        def detect_is_container() -> tuple[bool, str | None]:
            # Check custom properties / glTF extras on scene or objects
            prop_containers = [scene] + list(asset_objects) + [
                o.data for o in asset_objects if getattr(o, "data", None) is not None
            ]
            for holder in prop_containers:
                for key in holder.keys():
                    if key.startswith("_"):
                        continue
                    k_low = key.lower()
                    val = holder.get(key)
                    if k_low in {"is_container", "container"} and bool(val):
                        return True, f"custom_prop:{key}"
                    if k_low in {"tag", "tags", "category", "role", "type", "prop_type"}:
                        text = (
                            " ".join(str(x) for x in val)
                            if isinstance(val, (list, tuple, set))
                            else str(val)
                        )
                        tokens = tokenize_name(text)
                        if any(
                            t in CONTAINER_TAGS or t.endswith(tuple(CONTAINER_KEYWORDS))
                            for t in tokens
                        ):
                            return True, f"tag:{key}={text}"

            # Check asset ID, source filename, and object/mesh names
            names_to_check = [args.asset_id, src_path.stem] + [o.name for o in asset_objects]
            for nm in names_to_check:
                for tok in tokenize_name(nm):
                    if tok in CONTAINER_KEYWORDS or tok.endswith(tuple(CONTAINER_KEYWORDS)):
                        return True, f"name:{nm}"
            return False, None

        def measure_cavity_bounds(bounds: dict[str, Any]) -> dict[str, Any] | None:
            bmin = bounds["min"]
            bmax = bounds["max"]
            dims = bounds["dimensions_m"]
            dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
            if dx <= 1e-5 or dy <= 1e-5 or dz <= 1e-5:
                return None

            z_top = float(bmax[2])
            max_xy = math.hypot(dx, dy) + 0.05
            n_grid = 21
            n_rad = 24
            rad_dirs = [
                Vector((math.cos(2.0 * math.pi * k / n_rad), math.sin(2.0 * math.pi * k / n_rad), 0.0))
                for k in range(n_rad)
            ]

            def radial_hits_at(cx: float, cy: float, z: float) -> list[tuple[float, Vector]]:
                origin = Vector((cx, cy, z))
                hits: list[tuple[float, Vector]] = []
                for d in rad_dirs:
                    ok, loc, _norm, _idx, _obj, _mat = scene.ray_cast(
                        depsgraph, origin, d, distance=max_xy
                    )
                    if ok:
                        dist = math.hypot(loc.x - cx, loc.y - cy)
                        if dist > 1e-5:
                            hits.append((dist, loc))
                return hits

            # Step 1: 2D top-down grid to locate interior cavity floor candidates
            # (handles offset pans/pots with long handles or mug handles)
            min_cavity_depth = min(0.004, dz * 0.08)
            candidates: list[tuple[float, float, float, float]] = []

            for ix in range(1, n_grid - 1):
                gx = bmin[0] + (ix / (n_grid - 1)) * dx
                for iy in range(1, n_grid - 1):
                    gy = bmin[1] + (iy / (n_grid - 1)) * dy
                    ok, loc, _n, _idx, _o, _m = scene.ray_cast(
                        depsgraph,
                        Vector((gx, gy, z_top + 0.05)),
                        Vector((0.0, 0.0, -1.0)),
                        distance=dz + 0.2,
                    )
                    if not ok:
                        continue
                    floor_candidate_z = float(loc.z)
                    if (z_top - floor_candidate_z) < min_cavity_depth:
                        continue

                    # Confirm radial wall enclosure slightly above candidate floor
                    probe_z = floor_candidate_z + max(0.002, min(0.015, (z_top - floor_candidate_z) * 0.35))
                    hits = radial_hits_at(gx, gy, probe_z)
                    hit_frac = len(hits) / float(n_rad)
                    if hit_frac >= 0.75:
                        mean_r = sum(h[0] for h in hits) / len(hits)
                        score = hit_frac * 2.0 - (floor_candidate_z / max(dz, 1e-4)) * 0.5 + min(mean_r / max(dx, dy), 1.0) * 0.25
                        candidates.append((score, gx, gy, floor_candidate_z))

            if not candidates:
                return None

            candidates.sort(key=lambda item: item[0], reverse=True)
            _, cx, cy, approx_floor_z = candidates[0]

            # Step 2: Refine cavity center (cx, cy) using opposing radial wall midpoints
            refine_z = approx_floor_z + max(0.002, min(0.02, (z_top - approx_floor_z) * 0.4))
            for _ in range(2):
                origin = Vector((cx, cy, refine_z))
                pair_mids_x: list[float] = []
                pair_mids_y: list[float] = []
                half_rad = n_rad // 2
                for k in range(half_rad):
                    ok1, loc1, *_ = scene.ray_cast(depsgraph, origin, rad_dirs[k], distance=max_xy)
                    ok2, loc2, *_ = scene.ray_cast(
                        depsgraph, origin, rad_dirs[k + half_rad], distance=max_xy
                    )
                    if ok1 and ok2:
                        pair_mids_x.append(0.5 * (loc1.x + loc2.x))
                        pair_mids_y.append(0.5 * (loc1.y + loc2.y))
                if pair_mids_x:
                    cx = sum(pair_mids_x) / len(pair_mids_x)
                    cy = sum(pair_mids_y) / len(pair_mids_y)

            # Step 3: Exact floor_z via small cluster of downward rays near refined (cx, cy)
            floor_samples: list[float] = []
            probe_r = min(dx, dy) * 0.04
            offsets = [(0.0, 0.0)]
            for a in range(8):
                ang = 2.0 * math.pi * a / 8.0
                offsets.append((probe_r * math.cos(ang), probe_r * math.sin(ang)))
            for ox, oy in offsets:
                ok, loc, *_ = scene.ray_cast(
                    depsgraph,
                    Vector((cx + ox, cy + oy, z_top + 0.05)),
                    Vector((0.0, 0.0, -1.0)),
                    distance=dz + 0.2,
                )
                if ok:
                    floor_samples.append(float(loc.z))
            if not floor_samples:
                return None
            floor_z = min(floor_samples)
            if abs(floor_z) < 1e-6:
                floor_z = 0.0

            # Step 4: Sweep upward with radial rays to find true rim_z
            # (Ignores raised pan/pot handles because handles only block <20% of azimuths!)
            enclosure_threshold = 0.70
            n_slices = 48
            z_start = floor_z + max(0.001, (z_top - floor_z) * 0.02)
            if z_start >= z_top:
                return None

            last_enclosed_z: float | None = None
            first_open_z: float | None = None
            for i in range(n_slices + 1):
                z_cur = z_start + (i / n_slices) * (z_top - z_start)
                hits = radial_hits_at(cx, cy, z_cur)
                frac = len(hits) / float(n_rad)
                if frac >= enclosure_threshold:
                    last_enclosed_z = z_cur
                    first_open_z = None
                elif last_enclosed_z is not None and first_open_z is None:
                    first_open_z = z_cur
                    break

            if last_enclosed_z is None:
                return None

            if first_open_z is not None:
                lo_z, hi_z = last_enclosed_z, first_open_z
                for _ in range(10):
                    mid_z = 0.5 * (lo_z + hi_z)
                    hits = radial_hits_at(cx, cy, mid_z)
                    if (len(hits) / float(n_rad)) >= enclosure_threshold:
                        lo_z = mid_z
                    else:
                        hi_z = mid_z
                rim_z = lo_z
            else:
                rim_z = z_top

            depth_m = rim_z - floor_z
            if depth_m < min_cavity_depth:
                return None

            # Step 5: Measure inner_radius across lower, mid, and upper-rim interior heights
            def radius_at_z(z_val: float) -> float | None:
                hits = radial_hits_at(cx, cy, z_val)
                if len(hits) < max(6, n_rad // 2):
                    return None
                dists = sorted(h[0] for h in hits)
                mid_idx = len(dists) // 2
                return float(dists[mid_idx])

            r_bottom = radius_at_z(floor_z + 0.15 * depth_m)
            r_mid = radius_at_z(floor_z + 0.50 * depth_m)
            r_rim = radius_at_z(floor_z + 0.90 * depth_m)

            valid_radii = [r for r in (r_mid, r_rim, r_bottom) if r is not None]
            if not valid_radii:
                return None
            inner_radius = r_mid if r_mid is not None else valid_radii[0]

            return {
                "floor_z": round(float(floor_z), 6),
                "rim_z": round(float(rim_z), 6),
                "inner_radius": round(float(inner_radius), 6),
                "inner_radius_bottom": round(float(r_bottom), 6) if r_bottom is not None else None,
                "inner_radius_rim": round(float(r_rim), 6) if r_rim is not None else None,
                "depth_m": round(float(depth_m), 6),
                "cavity_center_xy": [round(float(cx), 6), round(float(cy), 6)],
            }

        is_container, container_reason = detect_is_container()
        meta["bottom_z_offset"] = round(float(asset_root.location.z), 6)
        meta["raw_min_z"] = round(float(pre_ground_min_z), 6)
        meta["materials"] = prop_materials
        meta["is_container"] = is_container
        if container_reason:
            meta["container_match_reason"] = container_reason
        if is_container:
            meta["cavity_bounds"] = measure_cavity_bounds(world_bounds)
        else:
            meta["cavity_bounds"] = None

    # ------------------------------------------------------------------
    # 4. Pack Textures & Save <id>_norm.blend (before adding temp rig)
    # ------------------------------------------------------------------
    packed_warnings: list[str] = []
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as exc:
        packed_warnings.append(str(exc))
        # Fallback: pack each valid external file individually
        for img in bpy.data.images:
            if img.source == "FILE" and img.packed_file is None:
                try:
                    img.pack()
                except RuntimeError as img_exc:
                    packed_warnings.append(f"{img.name}: {img_exc}")

    if packed_warnings:
        meta["pack_warnings"] = packed_warnings

    bpy.ops.wm.save_as_mainfile(
        filepath=str(norm_blend_path),
        check_existing=False,
        compress=True,
    )

    # ------------------------------------------------------------------
    # 5. Render Raw Views via BLENDER_EEVEE_NEXT
    # ------------------------------------------------------------------
    available_engines = {
        e.identifier
        for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    forced_engine = os.environ.get("BLENDER_PROFILE_ENGINE", "").strip().upper()
    headless_no_gpu = sys.platform == "linux" and not glob.glob("/dev/dri/renderD*")

    if forced_engine in available_engines:
        scene.render.engine = forced_engine
    elif headless_no_gpu and "CYCLES" in available_engines:
        scene.render.engine = "CYCLES"
    elif "BLENDER_EEVEE_NEXT" in available_engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in available_engines:
        scene.render.engine = "BLENDER_EEVEE"
    else:
        scene.render.engine = "CYCLES"

    if scene.render.engine == "CYCLES" and hasattr(scene, "cycles"):
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transmission_bounces = 2
        scene.cycles.transparent_max_bounces = 4
    elif hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 8
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = False
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = False
        if hasattr(scene.eevee, "use_shadows"):
            scene.eevee.use_shadows = False
    scene.render.use_persistent_data = False

    scene.render.resolution_x = PANEL_SIZE_PX
    scene.render.resolution_y = PANEL_SIZE_PX
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False

    # Neutral dark-slate studio world for clean silhouette contrast
    studio_world = bpy.data.worlds.new("ProfileStudioWorld")
    scene.world = studio_world
    studio_world.use_nodes = True
    w_nodes = studio_world.node_tree.nodes
    bg_node = next((n for n in w_nodes if n.type == "BACKGROUND"), None)
    if bg_node is not None:
        bg_node.inputs[0].default_value = (0.045, 0.050, 0.060, 1.0)
        bg_node.inputs[1].default_value = 0.85

    # Directional SUN lighting rig (scale-invariant across 5cm ramekins & 2m characters)
    def add_sun(name: str, direction_from: tuple[float, float, float], energy: float, color: tuple[float, float, float]) -> None:
        ldata = bpy.data.lights.new(name=name, type="SUN")
        ldata.energy = energy
        ldata.color = color
        ldata.angle = math.radians(15.0)
        lobj = bpy.data.objects.new(name=name, object_data=ldata)
        scene.collection.objects.link(lobj)
        vec = Vector(direction_from).normalized()
        # Sun shines along local -Z, so track -Z toward origin (align +Z to direction_from)
        lobj.rotation_euler = vec.to_track_quat("Z", "Y").to_euler()

    add_sun("KeySun", (1.2, -1.4, 1.6), 2.6, (1.0, 0.98, 0.95))
    add_sun("BackFillSun", (-1.1, 1.5, 1.0), 1.6, (0.90, 0.94, 1.0))
    add_sun("RimTopSun", (0.2, 0.8, 2.0), 0.9, (0.96, 0.98, 1.0))

    cam_data = bpy.data.cameras.new("ProfileCamera")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = 36.0
    cam_obj = bpy.data.objects.new("ProfileCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    b_center = Vector(world_bounds["center"])
    b_dims = Vector(world_bounds["dimensions_m"])
    cx, cy, cz = float(b_center.x), float(b_center.y), float(b_center.z)
    dx, dy, dz = float(b_dims.x), float(b_dims.y), float(b_dims.z)
    max_dim = max(dx, dy, dz, 0.02)
    ortho_dist = max_dim * 4.0 + 1.0

    def set_camera_basis(pos: Vector, right_vec: Vector, up_vec: Vector, forward_vec: Vector) -> None:
        """Set camera matrix_world from exact orthonormal world vectors.

        Blender camera local axes: +X = right, +Y = up, -Z = forward look direction.
        """
        r = right_vec.normalized()
        u = up_vec.normalized()
        neg_f = (-forward_vec).normalized()
        mat = Matrix((
            (r.x, u.x, neg_f.x, pos.x),
            (r.y, u.y, neg_f.y, pos.y),
            (r.z, u.z, neg_f.z, pos.z),
            (0.0, 0.0, 0.0, 1.0),
        ))
        cam_obj.matrix_world = mat

    def render_ortho_view(
        view_name: str,
        pos: Vector,
        right_vec: Vector,
        up_vec: Vector,
        forward_vec: Vector,
        ortho_scale_m: float,
        h_axis: str,
        v_axis: str,
        h_center_m: float,
        h_sign: float,
        v_center_m: float,
        v_sign: float,
    ) -> dict[str, Any]:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale_m
        cam_data.clip_start = max(0.001, ortho_dist * 0.01)
        cam_data.clip_end = ortho_dist * 4.0 + 10.0
        set_camera_basis(pos, right_vec, up_vec, forward_vec)

        out_png = raw_dir / f"{view_name}.png"
        scene.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True)

        half_m = ortho_scale_m * 0.5
        return {
            "raw_path": str(out_png),
            "projection": "ORTHO",
            "width_px": PANEL_SIZE_PX,
            "height_px": PANEL_SIZE_PX,
            "ortho_scale_m": round(ortho_scale_m, 8),
            "pixels_per_meter": round(PANEL_SIZE_PX / ortho_scale_m, 6),
            "h_axis": h_axis,
            "v_axis": v_axis,
            "h_left_m": round(h_center_m - h_sign * half_m, 8),
            "h_right_m": round(h_center_m + h_sign * half_m, 8),
            "v_bottom_m": round(v_center_m - v_sign * half_m, 8),
            "v_top_m": round(v_center_m + v_sign * half_m, 8),
        }

    def render_persp_view(view_name: str, view_dir_from_target: Vector, lens_mm: float = 50.0) -> dict[str, Any]:
        cam_data.type = "PERSP"
        cam_data.lens = lens_mm
        radius = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
        radius = max(radius, 0.01)
        half_fov = math.atan((cam_data.sensor_width * 0.5) / lens_mm)
        dist = (radius / math.sin(half_fov)) * 1.28

        cam_data.clip_start = max(0.001, dist * 0.01)
        cam_data.clip_end = dist * 8.0 + 10.0

        dir_norm = view_dir_from_target.normalized()
        pos = b_center + dir_norm * dist
        forward = (b_center - pos).normalized()
        world_up = Vector((0.0, 0.0, 1.0))
        right = forward.cross(world_up).normalized()
        up = right.cross(forward).normalized()
        set_camera_basis(pos, right, up, forward)

        out_png = raw_dir / f"{view_name}.png"
        scene.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True)
        return {
            "raw_path": str(out_png),
            "projection": "PERSP",
            "width_px": PANEL_SIZE_PX,
            "height_px": PANEL_SIZE_PX,
            "lens_mm": lens_mm,
        }

    render_views: dict[str, dict[str, Any]] = {}
    if args.category == "characters":
        # Shared orthographic scale across front/back/side so all 3 views match 1:1
        ortho_scale_char = max_dim * 1.34
        # Slight downward center padding so z=0.0 line has label breathing room
        char_v_center = cz - ortho_scale_char * 0.02

        render_views["front"] = render_ortho_view(
            view_name="front",
            pos=Vector((cx, cy - ortho_dist, char_v_center)),
            right_vec=Vector((1.0, 0.0, 0.0)),
            up_vec=Vector((0.0, 0.0, 1.0)),
            forward_vec=Vector((0.0, 1.0, 0.0)),
            ortho_scale_m=ortho_scale_char,
            h_axis="+X",
            v_axis="+Z",
            h_center_m=cx,
            h_sign=1.0,
            v_center_m=char_v_center,
            v_sign=1.0,
        )
        render_views["back"] = render_ortho_view(
            view_name="back",
            pos=Vector((cx, cy + ortho_dist, char_v_center)),
            right_vec=Vector((-1.0, 0.0, 0.0)),
            up_vec=Vector((0.0, 0.0, 1.0)),
            forward_vec=Vector((0.0, -1.0, 0.0)),
            ortho_scale_m=ortho_scale_char,
            h_axis="-X",
            v_axis="+Z",
            h_center_m=cx,
            h_sign=-1.0,
            v_center_m=char_v_center,
            v_sign=1.0,
        )
        render_views["side"] = render_ortho_view(
            view_name="side",
            pos=Vector((cx + ortho_dist, cy, char_v_center)),
            right_vec=Vector((0.0, 1.0, 0.0)),
            up_vec=Vector((0.0, 0.0, 1.0)),
            forward_vec=Vector((-1.0, 0.0, 0.0)),
            ortho_scale_m=ortho_scale_char,
            h_axis="+Y",
            v_axis="+Z",
            h_center_m=cy,
            h_sign=1.0,
            v_center_m=char_v_center,
            v_sign=1.0,
        )
        render_views["iso"] = render_persp_view(
            view_name="iso",
            view_dir_from_target=Vector((1.0, -1.2, 0.72)),
            lens_mm=50.0,
        )
    else:
        ortho_scale_top = max(dx, dy, 0.02) * 1.34
        render_views["three_quarter"] = render_persp_view(
            view_name="three_quarter",
            view_dir_from_target=Vector((0.9, -1.1, 0.78)),
            lens_mm=50.0,
        )
        render_views["top"] = render_ortho_view(
            view_name="top",
            pos=Vector((cx, cy, cz + ortho_dist)),
            right_vec=Vector((1.0, 0.0, 0.0)),
            up_vec=Vector((0.0, 1.0, 0.0)),
            forward_vec=Vector((0.0, 0.0, -1.0)),
            ortho_scale_m=ortho_scale_top,
            h_axis="+X",
            v_axis="+Y",
            h_center_m=cx,
            h_sign=1.0,
            v_center_m=cy,
            v_sign=1.0,
        )

    worker_payload = {
        "meta": meta,
        "render_views": render_views,
    }
    atomic_write_json(worker_json_path, worker_payload)
    return 0


# ===========================================================================
# OUTER PYTHON ORCHESTRATOR (Cache, Subprocess, Pillow Grids, Catalog)
# ===========================================================================


def discover_blender_executable(explicit_path: str | None = None) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified Blender executable not found: {p}")

    for env_key in ("BLENDER_BIN", "BLENDER_PATH", "BLENDER_EXECUTABLE"):
        env_val = os.environ.get(env_key)
        if env_val:
            p = Path(env_val).expanduser().resolve()
            if p.is_file():
                return p

    which_hit = shutil.which("blender")
    if which_hit:
        return Path(which_hit).resolve()

    candidates: list[str] = []
    sys_name = platform.system()
    if sys_name == "Windows":
        candidates.extend(
            [
                r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 4.5 LTS\blender.exe",
            ]
        )
        candidates.extend(
            sorted(
                glob.glob(r"C:\Program Files\Blender Foundation\Blender 4.*\blender.exe"),
                reverse=True,
            )
        )
    elif sys_name == "Darwin":
        candidates.extend(
            [
                "/Applications/Blender.app/Contents/MacOS/Blender",
                "/Applications/Blender 4.5.app/Contents/MacOS/Blender",
            ]
        )
        candidates.extend(
            sorted(
                glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender"),
                reverse=True,
            )
        )
    else:
        candidates.extend(
            [
                "/usr/local/bin/blender",
                "/usr/bin/blender",
                "/snap/bin/blender",
                str(Path.home() / "blender" / "blender"),
            ]
        )
        candidates.extend(
            sorted(
                glob.glob(str(Path.home() / "blender*" / "blender")),
                reverse=True,
            )
        )

    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return p

    raise FileNotFoundError(
        "Could not locate Blender 4.5 LTS executable. Pass --blender /path/to/blender "
        "or set BLENDER_BIN."
    )


def load_font(size: int) -> Any:
    from PIL import ImageFont  # type: ignore

    font_candidates = [
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "Consolas.ttf",
        "consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        r"C:\Windows\Fonts\consola.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_text_with_pill(
    draw: Any,
    xy: tuple[float, float],
    text: str,
    font: Any,
    text_fill: tuple[int, int, int, int],
    bg_fill: tuple[int, int, int, int] = (14, 17, 22, 200),
    pad: tuple[int, int] = (4, 2),
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - pad[0], bbox[1] - pad[1], bbox[2] + pad[0], bbox[3] + pad[1]),
        radius=3,
        fill=bg_fill,
    )
    draw.text((x, y), text, font=font, fill=text_fill)


def render_ortho_grid_overlay(
    raw_png_path: Path,
    view_info: dict[str, Any],
    panel_title: str,
    draw_ground_emphasis: bool = True,
) -> Any:
    """Draw labeled 0.1m X/Y/Z tick lines using Blender's exact orthographic bounds."""
    from PIL import Image, ImageDraw  # type: ignore

    base = Image.open(raw_png_path).convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_tick = load_font(12)
    font_badge = load_font(15)

    h_left = float(view_info["h_left_m"])
    h_right = float(view_info["h_right_m"])
    v_bottom = float(view_info["v_bottom_m"])
    v_top = float(view_info["v_top_m"])
    h_axis = str(view_info["h_axis"])
    v_axis = str(view_info["v_axis"])

    def axis_rgb(axis_label: str) -> tuple[int, int, int]:
        if "X" in axis_label:
            return (235, 95, 95)
        if "Y" in axis_label:
            return (95, 210, 120)
        return (95, 165, 240)

    h_rgb = axis_rgb(h_axis)
    v_rgb = axis_rgb(v_axis)

    span_h = abs(h_right - h_left)
    px_per_01m = (w / max(span_h, 1e-6)) * 0.1

    if px_per_01m >= 38:
        label_every = 1
    elif px_per_01m >= 18:
        label_every = 2
    elif px_per_01m >= 8:
        label_every = 5
    else:
        label_every = 10

    def world_h_to_px(val: float) -> float:
        return (val - h_left) / (h_right - h_left) * w

    def world_v_to_py(val: float) -> float:
        return (v_top - val) / (v_top - v_bottom) * h

    k_h_min = math.ceil(min(h_left, h_right) / 0.1)
    k_h_max = math.floor(max(h_left, h_right) / 0.1)
    for k in range(k_h_min, k_h_max + 1):
        val = k * 0.1
        px = world_h_to_px(val)
        if not (2 <= px <= w - 2):
            continue
        is_zero = k == 0
        is_major = (k % 5) == 0
        if is_zero:
            line_col = (*h_rgb, 150)
            line_w = 2
        elif is_major:
            line_col = (*h_rgb, 75)
            line_w = 1
        elif px_per_01m >= 12:
            line_col = (*h_rgb, 42)
            line_w = 1
        else:
            line_col = None
            line_w = 1

        if line_col is not None:
            draw.line([(px, 30), (px, h - 22)], fill=line_col, width=line_w)
        tick_len = 9 if (is_zero or is_major) else 5
        draw.line([(px, h - tick_len), (px, h)], fill=(*h_rgb, 220), width=line_w)

        if k % label_every == 0 and 24 <= px <= w - 34:
            label = f"{val:.1f}"
            draw_text_with_pill(
                draw,
                (px - 11, h - 22),
                label,
                font_tick,
                (*h_rgb, 245),
                bg_fill=(12, 15, 20, 195),
                pad=(3, 1),
            )

    k_v_min = math.ceil(min(v_bottom, v_top) / 0.1)
    k_v_max = math.floor(max(v_bottom, v_top) / 0.1)
    for k in range(k_v_min, k_v_max + 1):
        val = k * 0.1
        py = world_v_to_py(val)
        if not (2 <= py <= h - 2):
            continue
        is_zero = k == 0
        is_major = (k % 5) == 0
        if is_zero and draw_ground_emphasis and "Z" in v_axis:
            line_col = (248, 196, 70, 210)
            line_w = 2
            tick_col = (248, 196, 70, 255)
        elif is_zero:
            line_col = (*v_rgb, 150)
            line_w = 2
            tick_col = (*v_rgb, 230)
        elif is_major:
            line_col = (*v_rgb, 75)
            line_w = 1
            tick_col = (*v_rgb, 220)
        elif px_per_01m >= 12:
            line_col = (*v_rgb, 42)
            line_w = 1
            tick_col = (*v_rgb, 200)
        else:
            line_col = None
            line_w = 1
            tick_col = (*v_rgb, 180)

        if line_col is not None:
            draw.line([(46, py), (w - 6, py)], fill=line_col, width=line_w)
        tick_len = 10 if (is_zero or is_major) else 6
        draw.line([(0, py), (tick_len, py)], fill=tick_col, width=line_w)

        if (k % label_every == 0 or is_zero) and 32 <= py <= h - 28:
            lbl = "0.0 GND" if (is_zero and draw_ground_emphasis and "Z" in v_axis) else f"{val:.1f}"
            txt_col = (255, 215, 110, 255) if (is_zero and draw_ground_emphasis and "Z" in v_axis) else (*v_rgb, 245)
            draw_text_with_pill(
                draw,
                (12, py - 7),
                lbl,
                font_tick,
                txt_col,
                bg_fill=(12, 15, 20, 200),
                pad=(3, 1),
            )

    draw_text_with_pill(
        draw,
        (12, 10),
        panel_title,
        font_badge,
        (240, 244, 250, 255),
        bg_fill=(15, 18, 25, 215),
        pad=(6, 3),
    )
    draw_text_with_pill(
        draw,
        (w - 172, 10),
        f"H:{h_axis}  V:{v_axis}  [0.1m]",
        font_tick,
        (205, 215, 230, 240),
        bg_fill=(15, 18, 25, 205),
        pad=(5, 3),
    )

    return Image.alpha_composite(base, overlay).convert("RGB")


def render_clean_panel_badge(raw_png_path: Path, panel_title: str, note: str | None = None) -> Any:
    from PIL import Image, ImageDraw  # type: ignore

    base = Image.open(raw_png_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_badge = load_font(15)
    font_small = load_font(12)

    draw_text_with_pill(
        draw,
        (12, 10),
        panel_title,
        font_badge,
        (240, 244, 250, 255),
        bg_fill=(15, 18, 25, 215),
        pad=(6, 3),
    )
    if note:
        draw_text_with_pill(
            draw,
            (12, base.size[1] - 26),
            note,
            font_small,
            (205, 215, 230, 240),
            bg_fill=(15, 18, 25, 200),
            pad=(5, 2),
        )
    return Image.alpha_composite(base, overlay).convert("RGB")


def overlay_prop_cavity_on_top(
    img_rgb: Any, view_info: dict[str, Any], cavity_bounds: dict[str, Any] | None
) -> Any:
    """Visualize detected container cavity circle on the top ortho panel."""
    if not cavity_bounds or cavity_bounds.get("inner_radius") is None:
        return img_rgb
    from PIL import Image, ImageDraw  # type: ignore

    rgba = img_rgb.convert("RGBA")
    w, h = rgba.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    h_left = float(view_info["h_left_m"])
    h_right = float(view_info["h_right_m"])
    v_bottom = float(view_info["v_bottom_m"])
    v_top = float(view_info["v_top_m"])
    cx_m, cy_m = cavity_bounds["cavity_center_xy"]
    r_m = float(cavity_bounds["inner_radius"])

    px = (cx_m - h_left) / (h_right - h_left) * w
    py = (v_top - cy_m) / (v_top - v_bottom) * h
    rpx = r_m / abs(h_right - h_left) * w

    draw.ellipse(
        (px - rpx, py - rpx, px + rpx, py + rpx),
        outline=(80, 235, 245, 215),
        width=2,
    )
    draw.line([(px - 6, py), (px + 6, py)], fill=(80, 235, 245, 230), width=2)
    draw.line([(px, py - 6), (px, py + 6)], fill=(80, 235, 245, 230), width=2)

    font_small = load_font(12)
    draw_text_with_pill(
        draw,
        (12, h - 26),
        f"Cavity r={r_m:.3f}m  floor={cavity_bounds['floor_z']:.3f}m  rim={cavity_bounds['rim_z']:.3f}m",
        font_small,
        (120, 242, 250, 255),
        bg_fill=(12, 16, 22, 210),
        pad=(5, 2),
    )
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def stitch_character_grid(
    meta: dict[str, Any], render_views: dict[str, dict[str, Any]], out_png_path: Path
) -> None:
    from PIL import Image, ImageDraw  # type: ignore

    front_img = render_ortho_grid_overlay(
        Path(render_views["front"]["raw_path"]),
        render_views["front"],
        "FRONT (ORTHO)",
    )
    back_img = render_ortho_grid_overlay(
        Path(render_views["back"]["raw_path"]),
        render_views["back"],
        "BACK (ORTHO)",
    )
    side_img = render_ortho_grid_overlay(
        Path(render_views["side"]["raw_path"]),
        render_views["side"],
        "SIDE +X (ORTHO)",
    )
    iso_img = render_clean_panel_badge(
        Path(render_views["iso"]["raw_path"]),
        "ISO (3/4 PERSP)",
        note=f"Islands: {meta['loose_island_count']} | Bones: {len(meta['bones'])} | Anims: {len(meta['embedded_animations'])}",
    )

    pw = PANEL_SIZE_PX
    ph = PANEL_SIZE_PX
    sheet = Image.new("RGB", (pw * 2, CHARACTER_HEADER_PX + ph * 2), (16, 19, 25))
    sheet.paste(front_img, (0, CHARACTER_HEADER_PX))
    sheet.paste(back_img, (pw, CHARACTER_HEADER_PX))
    sheet.paste(side_img, (0, CHARACTER_HEADER_PX + ph))
    sheet.paste(iso_img, (pw, CHARACTER_HEADER_PX + ph))

    draw = ImageDraw.Draw(sheet)
    draw.line([(pw, CHARACTER_HEADER_PX), (pw, CHARACTER_HEADER_PX + ph * 2)], fill=(45, 52, 65), width=2)
    draw.line([(0, CHARACTER_HEADER_PX + ph), (pw * 2, CHARACTER_HEADER_PX + ph)], fill=(45, 52, 65), width=2)
    draw.line([(0, CHARACTER_HEADER_PX - 1), (pw * 2, CHARACTER_HEADER_PX - 1)], fill=(60, 70, 88), width=2)

    dx, dy, dz = meta["dimensions_m"]
    open_shells = sum(1 for s in meta["material_slots"] if not s["is_closed_shell"] and s["face_count"] > 0)
    closed_shells = sum(1 for s in meta["material_slots"] if s["is_closed_shell"])
    header_line1 = (
        f"CHARACTER: {meta['id']}   ({meta['source_file']})   "
        f"Dims: {dx:.3f} x {dy:.3f} x {dz:.3f} m"
    )
    header_line2 = (
        f"Islands: {meta['loose_island_count']}   "
        f"Bones: {len(meta['bones'])}   "
        f"Animations: {len(meta['embedded_animations'])}   "
        f"Slots: {len(meta['material_slots'])} (closed={closed_shells}, open={open_shells})"
    )
    draw.text((16, 8), header_line1, font=load_font(17), fill=(240, 245, 255))
    draw.text((16, 31), header_line2, font=load_font(14), fill=(165, 190, 225))

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png_path, format="PNG", optimize=True)


def stitch_prop_thumb(
    meta: dict[str, Any], render_views: dict[str, dict[str, Any]], out_png_path: Path
) -> None:
    from PIL import Image, ImageDraw  # type: ignore

    tq_img = render_clean_panel_badge(
        Path(render_views["three_quarter"]["raw_path"]),
        "3/4 (PERSP)",
    )
    top_img = render_ortho_grid_overlay(
        Path(render_views["top"]["raw_path"]),
        render_views["top"],
        "TOP (ORTHO)",
        draw_ground_emphasis=False,
    )
    if meta.get("is_container") and meta.get("cavity_bounds"):
        top_img = overlay_prop_cavity_on_top(top_img, render_views["top"], meta["cavity_bounds"])

    pw = PANEL_SIZE_PX
    ph = PANEL_SIZE_PX
    sheet = Image.new("RGB", (pw * 2, PROP_HEADER_PX + ph), (16, 19, 25))
    sheet.paste(tq_img, (0, PROP_HEADER_PX))
    sheet.paste(top_img, (pw, PROP_HEADER_PX))

    draw = ImageDraw.Draw(sheet)
    draw.line([(pw, PROP_HEADER_PX), (pw, PROP_HEADER_PX + ph)], fill=(45, 52, 65), width=2)
    draw.line([(0, PROP_HEADER_PX - 1), (pw * 2, PROP_HEADER_PX - 1)], fill=(60, 70, 88), width=2)

    dx, dy, dz = meta["dimensions_m"]
    mats_str = ", ".join(meta["materials"][:4])
    if len(meta["materials"]) > 4:
        mats_str += f" (+{len(meta['materials']) - 4})"

    header_line1 = (
        f"PROP: {meta['id']}   ({meta['source_file']})   "
        f"Dims: {dx:.3f} x {dy:.3f} x {dz:.3f} m   "
        f"bottom_z_offset: {meta['bottom_z_offset']:+.4f}m"
    )
    if meta.get("is_container"):
        cb = meta.get("cavity_bounds")
        if cb:
            cavity_str = (
                f"CONTAINER [floor_z={cb['floor_z']:.3f}m, rim_z={cb['rim_z']:.3f}m, "
                f"inner_r={cb['inner_radius']:.3f}m]"
            )
        else:
            cavity_str = "CONTAINER [cavity_bounds=none]"
    else:
        cavity_str = "Solid/Non-container"

    header_line2 = f"{cavity_str}   |   Materials ({len(meta['materials'])}): {mats_str or 'none'}"
    draw.text((16, 10), header_line1, font=load_font(17), fill=(240, 245, 255))
    draw.text((16, 37), header_line2, font=load_font(14), fill=(150, 215, 235) if meta.get("is_container") else (175, 185, 200))

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png_path, format="PNG", optimize=True)


def scan_source_assets(container_dir: Path) -> list[dict[str, Any]]:
    """Scan characters/ and props/ and assign deterministic unique asset IDs."""
    raw_entries: list[dict[str, Any]] = []
    for category in CATEGORIES:
        cat_dir = container_dir / category
        if not cat_dir.exists():
            continue
        files = sorted(
            (
                p
                for p in cat_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda p: p.name.lower(),
        )
        for fpath in files:
            stat = fpath.stat()
            rel_key = f"{category}/{fpath.name}"
            raw_entries.append(
                {
                    "rel_key": rel_key,
                    "category": category,
                    "path": fpath.resolve(),
                    "base_slug": sanitize_slug(fpath.stem),
                    "ext": fpath.suffix.lower(),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )

    slug_counts: dict[str, int] = {}
    for entry in raw_entries:
        slug_counts[entry["base_slug"]] = slug_counts.get(entry["base_slug"], 0) + 1

    used_ids: set[str] = set()
    for entry in raw_entries:
        candidate = entry["base_slug"]
        if slug_counts[candidate] > 1:
            cat_prefix = "char" if entry["category"] == "characters" else "prop"
            ext_tag = entry["ext"].lstrip(".")
            candidate = f"{cat_prefix}_{candidate}_{ext_tag}"
        final_id = candidate
        idx = 2
        while final_id in used_ids:
            final_id = f"{candidate}_{idx}"
            idx += 1
        used_ids.add(final_id)
        entry["id"] = final_id

    return raw_entries


def expected_sidecar_paths(profiles_dir: Path, asset_id: str, category: str) -> dict[str, Path]:
    preview_suffix = "_grid.png" if category == "characters" else "_thumb.png"
    return {
        "meta": profiles_dir / f"{asset_id}.meta.json",
        "blend": profiles_dir / f"{asset_id}_norm.blend",
        "preview": profiles_dir / f"{asset_id}{preview_suffix}",
    }


def remove_asset_sidecars(profiles_dir: Path, asset_id: str) -> list[str]:
    removed: list[str] = []
    for suffix in (".meta.json", "_norm.blend", "_grid.png", "_thumb.png"):
        p = profiles_dir / f"{asset_id}{suffix}"
        if p.exists():
            p.unlink()
            removed.append(p.name)
    return removed


def outer_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect, normalize, and render assets in import_container/ via headless Blender 4.5 LTS."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-inspect all assets even if size/mtime_ns and sidecars match cache.",
    )
    parser.add_argument(
        "--container-dir",
        default="import_container",
        help="Path to import_container directory (default: import_container).",
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Explicit path to Blender 4.5 LTS executable.",
    )
    args = parser.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print(
            "[ERROR] Pillow (PIL) is required in the outer Python environment. "
            "Install it with: pip install pillow",
            file=sys.stderr,
        )
        return 2

    container_dir = Path(args.container_dir).resolve()
    (container_dir / "characters").mkdir(parents=True, exist_ok=True)
    (container_dir / "props").mkdir(parents=True, exist_ok=True)

    index_dir = container_dir / "_index"
    profiles_dir = index_dir / "profiles"
    index_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    hashes_path = index_dir / "file_hashes.json"
    catalog_path = index_dir / "asset_catalog.json"

    existing_hashes: dict[str, dict[str, Any]] = {}
    if hashes_path.is_file():
        try:
            with hashes_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                existing_hashes = loaded
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Could not parse {hashes_path.name} ({exc}); rebuilding index.")

    scanned = scan_source_assets(container_dir)
    current_rel_keys = {e["rel_key"] for e in scanned}
    current_ids = {e["id"] for e in scanned}

    pruned_keys: list[str] = []
    for old_key, old_rec in list(existing_hashes.items()):
        if old_key not in current_rel_keys:
            pruned_keys.append(old_key)
            old_id = old_rec.get("id")
            if old_id and old_id not in current_ids:
                remove_asset_sidecars(profiles_dir, old_id)
            del existing_hashes[old_key]

    for p in list(profiles_dir.iterdir()):
        if not p.is_file():
            continue
        matched = False
        for aid in current_ids:
            if p.name in {
                f"{aid}.meta.json",
                f"{aid}_norm.blend",
                f"{aid}_grid.png",
                f"{aid}_thumb.png",
            }:
                matched = True
                break
        if not matched:
            p.unlink()

    to_process: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for entry in scanned:
        rel_key = entry["rel_key"]
        aid = entry["id"]
        sidecars = expected_sidecar_paths(profiles_dir, aid, entry["category"])
        cached = existing_hashes.get(rel_key)

        sidecars_exist = all(p.is_file() for p in sidecars.values())
        hash_matches = (
            cached is not None
            and cached.get("id") == aid
            and cached.get("size_bytes") == entry["size_bytes"]
            and cached.get("mtime_ns") == entry["mtime_ns"]
        )

        if not args.force and hash_matches and sidecars_exist:
            skipped.append(entry)
        else:
            to_process.append(entry)

    print(
        f"[profile_container] Scanned {len(scanned)} asset(s): "
        f"{len(to_process)} to inspect, {len(skipped)} cached/skipped, {len(pruned_keys)} pruned."
    )

    blender_bin: Path | None = None
    blender_version_str: str | None = None
    failures: list[tuple[str, str]] = []
    processed_count = 0

    if to_process:
        blender_bin = discover_blender_executable(args.blender)
        print(f"[profile_container] Using Blender: {blender_bin}")
        script_self = Path(__file__).resolve()

        for entry in to_process:
            rel_key = entry["rel_key"]
            aid = entry["id"]
            cat = entry["category"]
            sidecars = expected_sidecar_paths(profiles_dir, aid, cat)
            print(f"  -> Inspecting [{cat}] {entry['path'].name} as '{aid}' ...")

            alt_preview = profiles_dir / (
                f"{aid}_thumb.png" if cat == "characters" else f"{aid}_grid.png"
            )
            if alt_preview.exists():
                alt_preview.unlink()

            with tempfile.TemporaryDirectory(prefix=f"prof_{aid}_") as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                raw_dir = tmp_dir / "raw"
                worker_json = tmp_dir / "worker.json"

                cmd = [
                    str(blender_bin),
                    "--background",
                    "--factory-startup",
                    "-noaudio",
                    "--threads",
                    "2",
                    "--python",
                    str(script_self),
                    "--",
                    "--blender-worker",
                    "--src",
                    str(entry["path"]),
                    "--category",
                    cat,
                    "--asset-id",
                    aid,
                    "--norm-blend",
                    str(sidecars["blend"]),
                    "--raw-dir",
                    str(raw_dir),
                    "--worker-json",
                    str(worker_json),
                ]

                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                if proc.returncode != 0 or not worker_json.is_file():
                    err_tail = "\n".join(proc.stdout.splitlines()[-25:])
                    failures.append((rel_key, f"exit={proc.returncode}\n{err_tail}"))
                    print(f"  [ERROR] Failed {rel_key}:\n{err_tail}", file=sys.stderr)
                    continue

                with worker_json.open("r", encoding="utf-8") as fh:
                    worker_data = json.load(fh)

                meta = worker_data["meta"]
                render_views = worker_data["render_views"]
                blender_version_str = meta.get("blender_version") or blender_version_str

                if cat == "characters":
                    stitch_character_grid(meta, render_views, sidecars["preview"])
                else:
                    stitch_prop_thumb(meta, render_views, sidecars["preview"])

                clean_bounds = {
                    k: {vk: vv for vk, vv in v.items() if vk != "raw_path"}
                    for k, v in render_views.items()
                }
                meta["ortho_bounds"] = clean_bounds
                meta["files"] = {
                    "source": str(entry["path"].relative_to(container_dir)).replace("\\", "/"),
                    "norm_blend": str(sidecars["blend"].relative_to(container_dir)).replace("\\", "/"),
                    "preview_image": str(sidecars["preview"].relative_to(container_dir)).replace("\\", "/"),
                    "meta_json": str(sidecars["meta"].relative_to(container_dir)).replace("\\", "/"),
                }
                meta["inspected_at"] = datetime.now(timezone.utc).isoformat()

                atomic_write_json(sidecars["meta"], meta)

                existing_hashes[rel_key] = {
                    "id": aid,
                    "category": cat,
                    "size_bytes": entry["size_bytes"],
                    "mtime_ns": entry["mtime_ns"],
                    "meta_file": meta["files"]["meta_json"],
                    "norm_blend": meta["files"]["norm_blend"],
                    "preview_file": meta["files"]["preview_image"],
                    "updated_at": meta["inspected_at"],
                }
                processed_count += 1
                print(
                    f"     OK dims={meta['dimensions_m']} "
                    f"-> {sidecars['preview'].name}, {sidecars['blend'].name}"
                )

    characters_catalog: dict[str, Any] = {}
    props_catalog: dict[str, Any] = {}
    all_assets_summary: list[dict[str, Any]] = []

    for entry in scanned:
        aid = entry["id"]
        cat = entry["category"]
        sidecars = expected_sidecar_paths(profiles_dir, aid, cat)
        if not sidecars["meta"].is_file():
            continue
        with sidecars["meta"].open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if blender_version_str is None:
            blender_version_str = meta.get("blender_version")

        if cat == "characters":
            summary = {
                "id": aid,
                "category": cat,
                "source_file": meta.get("source_file"),
                "files": meta.get("files"),
                "dimensions_m": meta.get("dimensions_m"),
                "bounds_m": meta.get("bounds_m"),
                "loose_island_count": meta.get("loose_island_count"),
                "bones": meta.get("bones", []),
                "embedded_animations": meta.get("embedded_animations", []),
                "material_slots": meta.get("material_slots", []),
            }
            characters_catalog[aid] = summary
        else:
            summary = {
                "id": aid,
                "category": cat,
                "source_file": meta.get("source_file"),
                "files": meta.get("files"),
                "dimensions_m": meta.get("dimensions_m"),
                "bounds_m": meta.get("bounds_m"),
                "bottom_z_offset": meta.get("bottom_z_offset"),
                "materials": meta.get("materials", []),
                "is_container": meta.get("is_container", False),
                "cavity_bounds": meta.get("cavity_bounds"),
            }
            props_catalog[aid] = summary

        all_assets_summary.append(summary)

    catalog_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blender_version": blender_version_str,
        "container_dir": str(container_dir),
        "counts": {
            "total": len(all_assets_summary),
            "characters": len(characters_catalog),
            "props": len(props_catalog),
        },
        "characters": characters_catalog,
        "props": props_catalog,
        "assets": all_assets_summary,
    }

    atomic_write_json(hashes_path, existing_hashes)
    atomic_write_json(catalog_path, catalog_payload)

    print(
        f"[profile_container] Done: processed={processed_count}, skipped={len(skipped)}, "
        f"pruned={len(pruned_keys)}, failed={len(failures)} | "
        f"Catalog: {catalog_path}"
    )

    return 1 if failures else 0


# ===========================================================================
# ENTRYPOINT DISPATCH
# ===========================================================================


def main() -> int:
    if "--blender-worker" in sys.argv:
        if "--" in sys.argv:
            after_sep = sys.argv[sys.argv.index("--") + 1 :]
        else:
            after_sep = sys.argv[1:]
        worker_argv = [arg for arg in after_sep if arg != "--blender-worker"]
        return blender_worker_main(worker_argv)
    return outer_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())