"""
scripts/bpy_ops_kit.py — Blocks 2.2–2.4 Blender 4.5 LTS Helper Library

Role:
    Imported directly inside Hydrogen's generated Blender Python scripts
    (`import bpy_ops_kit`). Provides verified, headless-safe (`blender -b`)
    operations strictly targeting Blender 4.5 LTS APIs and built-in modules
    (bpy, bmesh, mathutils, math, json, os).

Modules:
    1. Setup & Safe Loading
    2. Sanitization & The 6 Universal Geometry Operations (2.2:2)
    3. Blender 4.5 Procedural & X-Ray Shaders
    4. Staging, Camera, & Animation Helpers (2.3:1 & 2.4:1)
    5. Rapid Preview Rendering (2.2:3, 2.3:2, 2.4:2)
"""

import json
import math
import os

import bmesh
import bpy
import mathutils


# ==============================================================================
# INTERNAL UTILITIES (PURE BPY / BMESH / MATHUTILS)
# ==============================================================================

def _to_rgba(color, default_alpha=1.0):
    """Normalize a 3- or 4-component color tuple/Vector to (r, g, b, a)."""
    if color is None:
        return (0.8, 0.8, 0.8, float(default_alpha))
    vals = [float(c) for c in color]
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2], float(default_alpha))
    if len(vals) >= 4:
        return (vals[0], vals[1], vals[2], vals[3])
    raise ValueError(f"Expected RGB or RGBA color sequence, got: {color!r}")


def _ensure_collection(name, parent=None):
    """Get or create a Blender Collection and ensure it is linked to parent."""
    if parent is None:
        parent = bpy.context.scene.collection
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def _link_object_to_collections(obj, collections):
    """Safely link an object to one or more collections (fallback: scene root)."""
    target_colls = list(collections) if collections else [bpy.context.scene.collection]
    for coll in target_colls:
        if obj.name not in coll.objects:
            coll.objects.link(obj)


def _resolve_mesh_object(obj):
    """Resolve a MESH object directly or from a single-mesh Empty wrapper."""
    if obj is None:
        raise ValueError("Expected a Blender Object, got None.")
    if obj.type == "MESH":
        return obj
    if obj.type == "EMPTY":
        mesh_children = [c for c in obj.children_recursive if c.type == "MESH"]
        if len(mesh_children) == 1:
            return mesh_children[0]
        raise ValueError(
            f"Empty '{obj.name}' contains {len(mesh_children)} mesh children; "
            "pass the specific mesh object directly."
        )
    raise TypeError(f"Object '{obj.name}' has type '{obj.type}'; expected 'MESH'.")


def _find_material_indices(obj, material_name):
    """Find material slot indices matching material_name (exact, .00x, or CI)."""
    if not material_name:
        return []
    slots = obj.material_slots
    exact = [i for i, s in enumerate(slots) if s.material and s.material.name == material_name]
    if exact:
        return exact
    prefix = material_name + "."
    suffixed = [i for i, s in enumerate(slots) if s.material and s.material.name.startswith(prefix)]
    if suffixed:
        return suffixed
    target_low = material_name.lower()
    return [i for i, s in enumerate(slots) if s.material and s.material.name.lower() == target_low]


def _copy_used_materials(src_materials, bm, dst_mesh):
    """Copy only materials referenced by bm.faces onto dst_mesh and remap indices."""
    used_indices = sorted({f.material_index for f in bm.faces if 0 <= f.material_index < len(src_materials)})
    index_map = {}
    for old_idx in used_indices:
        mat = src_materials[old_idx]
        if mat is not None:
            new_idx = len(dst_mesh.materials)
            dst_mesh.materials.append(mat)
            index_map[old_idx] = new_idx
    for f in bm.faces:
        f.material_index = index_map.get(f.material_index, 0)


def _find_bm_face_islands(bm):
    """Decompose bm.faces into connected islands via shared vertices (BFS)."""
    bm.faces.ensure_lookup_table()
    visited = set()
    islands = []
    for f in bm.faces:
        if f.index in visited:
            continue
        comp = []
        queue = [f]
        visited.add(f.index)
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            comp.append(cur)
            for v in cur.verts:
                for nf in v.link_faces:
                    if nf.index not in visited:
                        visited.add(nf.index)
                        queue.append(nf)
        islands.append(comp)
    return islands


def _is_face_subset_closed(faces):
    """Return True iff every edge incident to `faces` has exactly 2 faces in `faces`."""
    if not faces:
        return False
    face_set = set(faces)
    checked_edges = set()
    for f in faces:
        for e in f.edges:
            if e in checked_edges:
                continue
            checked_edges.add(e)
            if sum(1 for lf in e.link_faces if lf in face_set) != 2:
                return False
    return True


def _apply_planar_uvs(bm, faces, plane_no):
    """Project metric planar UVs onto newly created cap faces for all UV layers."""
    if not faces or not bm.loops.layers.uv:
        return
    n = plane_no.normalized() if plane_no.length > 1e-8 else mathutils.Vector((0.0, 0.0, 1.0))
    ref = mathutils.Vector((1.0, 0.0, 0.0)) if abs(n.x) < 0.9 else mathutils.Vector((0.0, 1.0, 0.0))
    u_axis = n.cross(ref).normalized()
    v_axis = n.cross(u_axis).normalized()
    for uv_layer in bm.loops.layers.uv.values():
        for f in faces:
            if not f.is_valid:
                continue
            for loop in f.loops:
                co = loop.vert.co
                loop[uv_layer].uv = (co.dot(u_axis), co.dot(v_axis))


def _fill_boundary_loops(
    bm,
    boundary_edges,
    interior_mat_idx=None,
    expected_outward_no=None,
    smooth_cap=False,
):
    """Cap boundary loops cleanly via holes_fill + triangle_fill fallback."""
    valid_edges = [e for e in boundary_edges if e.is_valid and e.is_boundary]
    if not valid_edges:
        return []

    prev_faces = {f for f in bm.faces if f.is_valid}

    try:
        bmesh.ops.holes_fill(bm, edges=valid_edges, sides=0)
    except Exception:
        pass

    rem_edges = [e for e in valid_edges if e.is_valid and e.is_boundary]
    if rem_edges:
        try:
            bmesh.ops.triangle_fill(bm, edges=rem_edges, use_beauty=True)
        except Exception:
            pass

    new_faces = [f for f in bm.faces if f.is_valid and f not in prev_faces]
    for f in new_faces:
        f.smooth = bool(smooth_cap)
        if interior_mat_idx is not None:
            f.material_index = int(interior_mat_idx)

    if not smooth_cap:
        for e in valid_edges:
            if e.is_valid:
                e.smooth = False

    if expected_outward_no is not None and expected_outward_no.length > 1e-8:
        _apply_planar_uvs(bm, new_faces, expected_outward_no)
    elif new_faces:
        _apply_planar_uvs(bm, new_faces, new_faces[0].normal)

    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    if expected_outward_no is not None and expected_outward_no.length > 1e-8:
        out_n = expected_outward_no.normalized()
        for f in new_faces:
            if f.is_valid and f.normal.dot(out_n) < 0.0:
                f.normal_flip()

    return new_faces


def _plane_from_args(plane_co=None, plane_no=None, axis=None, threshold=None):
    """Resolve (plane_co, plane_no) from explicit vectors or (axis, threshold)."""
    if plane_co is not None and plane_no is not None:
        co = mathutils.Vector(plane_co)
        no = mathutils.Vector(plane_no)
        if no.length < 1e-8:
            raise ValueError("plane_no must be non-zero.")
        return co, no.normalized()

    if axis is not None:
        ax = str(axis).strip().upper()
        sign = 1.0
        if ax.startswith("-"):
            sign = -1.0
            ax = ax[1:]
        elif ax.startswith("+"):
            ax = ax[1:]
        axis_map = {
            "X": mathutils.Vector((sign, 0.0, 0.0)),
            "Y": mathutils.Vector((0.0, sign, 0.0)),
            "Z": mathutils.Vector((0.0, 0.0, sign)),
        }
        if ax not in axis_map:
            raise ValueError(f"Unsupported axis '{axis}'; expected X, Y, Z, -X, -Y, or -Z.")
        t = 0.0 if threshold is None else float(threshold)
        co = mathutils.Vector((
            t if ax == "X" else 0.0,
            t if ax == "Y" else 0.0,
            t if ax == "Z" else 0.0,
        ))
        return co, axis_map[ax]

    co = mathutils.Vector((0.0, 0.0, 0.0)) if plane_co is None else mathutils.Vector(plane_co)
    no = mathutils.Vector((0.0, 0.0, 1.0)) if plane_no is None else mathutils.Vector(plane_no).normalized()
    return co, no


def _world_plane_to_local(obj, plane_co, plane_no):
    """Transform a world-space plane (point, normal) into obj's local space."""
    inv = obj.matrix_world.inverted()
    local_co = inv @ mathutils.Vector(plane_co)
    local_no = obj.matrix_world.to_3x3().transposed() @ mathutils.Vector(plane_no)
    if local_no.length < 1e-10:
        raise ValueError(f"Object '{obj.name}' has a singular world matrix.")
    return local_co, local_no.normalized()


def _copy_transform_hierarchy(src_obj, dst_obj):
    """Copy local transform, rotation mode, parent, and parent inverse."""
    dst_obj.rotation_mode = src_obj.rotation_mode
    dst_obj.location = src_obj.location.copy()
    if src_obj.rotation_mode == "QUATERNION":
        dst_obj.rotation_quaternion = src_obj.rotation_quaternion.copy()
    elif src_obj.rotation_mode == "AXIS_ANGLE":
        dst_obj.rotation_axis_angle = tuple(src_obj.rotation_axis_angle)
    else:
        dst_obj.rotation_euler = src_obj.rotation_euler.copy()
    dst_obj.scale = src_obj.scale.copy()
    if src_obj.parent is not None:
        dst_obj.parent = src_obj.parent
        dst_obj.matrix_parent_inverse = src_obj.matrix_parent_inverse.copy()


def _create_chunk_object(
    bm,
    src_obj,
    name,
    src_materials=None,
    center_pivot=True,
    collections=None,
):
    """Write a BMesh into a new Object inheriting src_obj transforms/materials."""
    mesh = bpy.data.meshes.new(name + "_Mesh")

    if center_pivot and len(bm.verts) > 0:
        local_centroid = sum((v.co for v in bm.verts), mathutils.Vector((0.0, 0.0, 0.0))) / len(bm.verts)
        for v in bm.verts:
            v.co -= local_centroid
    else:
        local_centroid = mathutils.Vector((0.0, 0.0, 0.0))

    mats = src_materials if src_materials is not None else [s.material for s in src_obj.material_slots]
    _copy_used_materials(mats, bm, mesh)

    bm.to_mesh(mesh)
    mesh.update()

    new_obj = bpy.data.objects.new(name, mesh)
    _copy_transform_hierarchy(src_obj, new_obj)

    if center_pivot and local_centroid.length > 0.0:
        new_obj.location = src_obj.location + (src_obj.matrix_basis.to_3x3() @ local_centroid)

    target_colls = collections if collections is not None else src_obj.users_collection
    _link_object_to_collections(new_obj, target_colls)
    return new_obj


def _make_lcg(seed):
    """Deterministic pure-Python LCG returning floats in [0.0, 1.0)."""
    state = [(int(seed) & 0xFFFFFFFF) or 1]

    def _rand():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296.0

    return _rand


def _ensure_interior_material(obj_or_mats, interior_material=None, interior_material_name="Cross_Section_Interior"):
    """Return (material, slot_index) for a slice/fracture interior material."""
    if interior_material is not None:
        mat = interior_material
    else:
        mat = bpy.data.materials.get(interior_material_name)
        if mat is None:
            mat = make_pbr_material(
                name=interior_material_name,
                base_color=(0.78, 0.56, 0.42, 1.0),
                secondary_color=(0.64, 0.42, 0.30, 1.0),
                roughness=0.55,
                roughness_variation=0.10,
                pattern="VORONOI",
                pattern_scale=18.0,
                bump_strength=0.12,
                subsurface_weight=0.05,
            )

    if isinstance(obj_or_mats, bpy.types.Object):
        mats = [s.material for s in obj_or_mats.material_slots]
    else:
        mats = list(obj_or_mats)

    for idx, m in enumerate(mats):
        if m == mat or (m is not None and m.name == mat.name):
            return mat, idx

    mats.append(mat)
    return mat, len(mats) - 1


def _safe_set_bsdf_input(bsdf, name, value):
    """Safely assign a Principled BSDF socket default_value if present."""
    sock = bsdf.inputs.get(name)
    if sock is not None:
        sock.default_value = value


# ==============================================================================
# MODULE 1 — SETUP & SAFE LOADING
# ==============================================================================

def clear_scene(keep_cameras=False, keep_lights=False):
    """Remove scene objects and purge orphaned datablocks cleanly."""
    for obj in list(bpy.data.objects):
        if keep_cameras and obj.type == "CAMERA":
            continue
        if keep_lights and obj.type == "LIGHT":
            continue
        bpy.data.objects.remove(obj, do_unlink=True)

    for coll in list(bpy.data.collections):
        if len(coll.objects) == 0 and len(coll.children) == 0:
            bpy.data.collections.remove(coll)

    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def get_world_bounds(obj_or_objs):
    """
    Compute exact world-space AABB (min_vec, max_vec) across one or more
    Blender objects (recursively including MESH children for Empties/Armatures).
    """
    if isinstance(obj_or_objs, (list, tuple, set)):
        roots = list(obj_or_objs)
    else:
        roots = [obj_or_objs]

    candidates = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        for o in [root] + list(root.children_recursive):
            if o.name in seen:
                continue
            seen.add(o.name)
            if o.type == "MESH" and o.data and len(o.data.vertices) > 0:
                candidates.append(o)

    if not candidates:
        candidates = [o for o in roots if o is not None and o.type != "EMPTY"]

    if not candidates:
        pos = roots[0].matrix_world.translation.copy() if roots and roots[0] else mathutils.Vector((0.0, 0.0, 0.0))
        return pos.copy(), pos.copy()

    bpy.context.view_layer.update()

    inf = float("inf")
    min_v = mathutils.Vector((inf, inf, inf))
    max_v = mathutils.Vector((-inf, -inf, -inf))

    for obj in candidates:
        mat = obj.matrix_world
        for corner in obj.bound_box:
            wc = mat @ mathutils.Vector(corner)
            min_v.x = min(min_v.x, wc.x)
            min_v.y = min(min_v.y, wc.y)
            min_v.z = min(min_v.z, wc.z)
            max_v.x = max(max_v.x, wc.x)
            max_v.y = max(max_v.y, wc.y)
            max_v.z = max(max_v.z, wc.z)

    return min_v, max_v


def seat_on_z(obj, target_z=0.0):
    """Translate `obj` in world Z so its world bound_box min Z sits on target_z."""
    min_v, _ = get_world_bounds(obj)
    obj.location.z += float(target_z) - min_v.z
    bpy.context.view_layer.update()
    return obj


def configure_eevee_next(
    fps=30,
    samples=16,
    width=1080,
    height=1920,
    view_transform="AgX",
    look="AgX - Medium High Contrast",
    exposure=0.0,
    use_raytracing=True,
    transparent_bg=False,
):
    """
    Lock Blender 4.5 EEVEE Next ('BLENDER_EEVEE_NEXT'), portrait 1080x1920
    resolution, PNG output, and AgX color management.
    """
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"

    scene.render.fps = int(fps)
    scene.render.fps_base = 1.0

    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    scene.render.film_transparent = bool(transparent_bg)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"
    scene.render.image_settings.color_depth = "8"

    eevee = scene.eevee
    eevee.taa_render_samples = int(samples)
    if hasattr(eevee, "taa_samples"):
        eevee.taa_samples = min(int(samples), 16)
    if hasattr(eevee, "use_raytracing"):
        eevee.use_raytracing = bool(use_raytracing)
    if hasattr(eevee, "use_shadows"):
        eevee.use_shadows = True
    if hasattr(eevee, "shadow_resolution_scale"):
        eevee.shadow_resolution_scale = 1.0

    scene.display_settings.display_device = "sRGB"
    try:
        scene.view_settings.view_transform = view_transform
    except TypeError:
        scene.view_settings.view_transform = "Standard"

    try:
        scene.view_settings.look = look
    except TypeError:
        scene.view_settings.look = "None"

    scene.view_settings.exposure = float(exposure)
    scene.view_settings.gamma = 1.0
    return scene


def append_from_blend(
    blend_path,
    names=None,
    parent_name=None,
    target_height_m=None,
    ground_to_z=0.0,
    collection_name=None,
):
    """
    Headless-safe append via `bpy.data.libraries.load` from `_norm.blend` or
    `2_assets/models/*.blend`.

    Imports matching collections or objects, wraps them inside an Empty
    (`parent_name`), uniformly scales to `target_height_m`, and seats world
    bound_box min Z onto `ground_to_z`.
    """
    abs_path = os.path.abspath(blend_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Blend file not found: {abs_path}")

    if isinstance(names, str):
        requested = [names]
    elif names is not None:
        requested = list(names)
    else:
        requested = None

    with bpy.data.libraries.load(abs_path, link=False) as (data_from, data_to):
        avail_colls = set(data_from.collections)
        avail_objs = set(data_from.objects)

        if requested is None:
            data_to.objects = list(data_from.objects)
        else:
            colls_to_load = [n for n in requested if n in avail_colls]
            objs_to_load = [n for n in requested if n in avail_objs and n not in avail_colls]
            if not colls_to_load and not objs_to_load:
                raise ValueError(
                    f"None of {requested} found in '{abs_path}'. "
                    f"Available collections={sorted(avail_colls)}, objects={sorted(avail_objs)}"
                )
            data_to.collections = colls_to_load
            data_to.objects = objs_to_load

    loaded_colls = [c for c in data_to.collections if c is not None]
    loaded_objs = [o for o in data_to.objects if o is not None]

    target_coll = (
        _ensure_collection(collection_name)
        if collection_name
        else bpy.context.scene.collection
    )

    all_imported = []
    seen = set()

    for coll in loaded_colls:
        if coll.name not in target_coll.children:
            target_coll.children.link(coll)
        for o in coll.all_objects:
            if o.name not in seen:
                seen.add(o.name)
                all_imported.append(o)

    for o in loaded_objs:
        if o.name not in seen:
            seen.add(o.name)
            all_imported.append(o)
        if o.name not in target_coll.objects and not any(o.name in c.all_objects for c in loaded_colls):
            target_coll.objects.link(o)

    if not parent_name:
        base = os.path.splitext(os.path.basename(abs_path))[0]
        parent_name = f"{base}_Root"

    root_empty = bpy.data.objects.new(parent_name, None)
    root_empty.empty_display_type = "PLAIN_AXES"
    root_empty.empty_display_size = 0.2
    target_coll.objects.link(root_empty)

    imported_set = set(all_imported)
    for o in all_imported:
        if o.parent is None or o.parent not in imported_set:
            o.parent = root_empty
            o.matrix_parent_inverse.identity()

    bpy.context.view_layer.update()

    if target_height_m is not None:
        min_v, max_v = get_world_bounds(root_empty)
        cur_height = max_v.z - min_v.z
        if cur_height > 1e-6:
            root_empty.scale = root_empty.scale * (float(target_height_m) / cur_height)
            bpy.context.view_layer.update()

    if ground_to_z is not None:
        seat_on_z(root_empty, target_z=float(ground_to_z))

    root_empty["imported_objects_json"] = json.dumps([o.name for o in all_imported])
    return root_empty


# ==============================================================================
# MODULE 2 — SANITIZATION & THE 6 UNIVERSAL GEOMETRY OPERATIONS (2.2:2)
# ==============================================================================

def sanitize_for_surgery(obj, merge_dist=1e-5, freeze_transform=False):
    """
    Freeze active poses/modifiers (`convert(target='MESH')`), weld split
    UV-seam vertices (`bmesh.ops.remove_doubles`), clean orphaned vertices,
    and recalculate outward normals.
    """
    if obj is None:
        raise ValueError("sanitize_for_surgery received None.")

    if obj.type == "EMPTY":
        for child in list(obj.children_recursive):
            if child.type in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
                sanitize_for_surgery(child, merge_dist=merge_dist, freeze_transform=freeze_transform)
        return obj

    if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META"):
        return obj

    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    if len(obj.users_collection) == 0:
        bpy.context.scene.collection.objects.link(obj)

    was_hidden_vp = obj.hide_viewport
    obj.hide_viewport = False
    bpy.context.view_layer.update()

    try:
        try:
            with bpy.context.temp_override(active_object=obj, selected_objects=[obj], object=obj):
                bpy.ops.object.convert(target="MESH")
        except RuntimeError:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            new_mesh = bpy.data.meshes.new_from_object(
                eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph
            )
            old_mesh = obj.data
            obj.modifiers.clear()
            obj.data = new_mesh
            if old_mesh and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
    finally:
        obj.hide_viewport = was_hidden_vp

    if obj.type != "MESH" or obj.data is None:
        return obj

    if freeze_transform:
        saved_world = obj.matrix_world.copy()
        obj.data.transform(saved_world)
        obj.matrix_world.identity()

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    if len(bm.verts) > 0 and merge_dist is not None and merge_dist > 0.0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=float(merge_dist))

    orphan_verts = [v for v in bm.verts if not v.link_faces and not v.link_edges]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context="VERTS")

    if len(bm.faces) > 0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def detach_part(
    obj,
    part_name="Detached_Part",
    material_name=None,
    plane_co=None,
    plane_no=None,
    axis=None,
    threshold=None,
    keep_side="POSITIVE",
    coord_space="WORLD",
    shell_thickness=0.008,
    solidify_offset=-1.0,
    cap_wound=True,
    cap_material_name=None,
    apply_solidify=False,
):
    """
    Universal Op 1: Isolate faces by `material_name` and/or a metric coordinate
    plane (`plane_co`/`plane_no` or `axis`/`threshold`), separate into a new
    object with SOLIDIFY shell thickness, and cap the host wound cleanly.
    """
    host = _resolve_mesh_object(obj)
    if len(host.modifiers) > 0:
        sanitize_for_surgery(host)

    has_plane = (plane_co is not None and plane_no is not None) or (axis is not None)
    if material_name is None and not has_plane:
        raise ValueError("detach_part requires `material_name` or a cutting plane (`axis`/`threshold` or `plane_co`/`plane_no`).")

    bm = bmesh.new()
    bm.from_mesh(host.data)

    local_co = local_no = None
    if has_plane:
        co, no = _plane_from_args(plane_co=plane_co, plane_no=plane_no, axis=axis, threshold=threshold)
        if str(coord_space).upper() == "WORLD":
            local_co, local_no = _world_plane_to_local(host, co, no)
        else:
            local_co, local_no = co, no

        bmesh.ops.bisect_plane(
            bm,
            geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            plane_co=local_co,
            plane_no=local_no,
            dist=1e-5,
            clear_inner=False,
            clear_outer=False,
        )

    # Re-index after bisect_plane so copied BMesh indices match 1-to-1
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    mat_indices = None
    if material_name is not None:
        mat_indices = set(_find_material_indices(host, material_name))
        if not mat_indices:
            bm.free()
            avail = [s.material.name for s in host.material_slots if s.material]
            raise ValueError(f"Material '{material_name}' not found on '{host.name}'. Available: {avail}")

    side_pos = str(keep_side).strip().upper() != "NEGATIVE"
    isolated_faces = []
    for f in bm.faces:
        if mat_indices is not None and f.material_index not in mat_indices:
            continue
        if has_plane:
            signed = (f.calc_center_median() - local_co).dot(local_no)
            if side_pos and signed < -1e-6:
                continue
            if not side_pos and signed > 1e-6:
                continue
        isolated_faces.append(f)

    if not isolated_faces:
        bm.free()
        raise ValueError(f"No faces matched detachment criteria on '{host.name}'.")

    isolated_set = set(isolated_faces)
    if len(isolated_set) == len(bm.faces):
        bm.free()
        raise ValueError(f"Detachment criteria selected 100% of faces on '{host.name}', leaving no host mesh.")

    seam_vert_indices = set()
    for e in bm.edges:
        has_iso = any(f in isolated_set for f in e.link_faces)
        has_host = any(f not in isolated_set for f in e.link_faces)
        if has_iso and has_host:
            seam_vert_indices.add(e.verts[0].index)
            seam_vert_indices.add(e.verts[1].index)

    iso_indices = {f.index for f in isolated_faces}
    detached_bm = bm.copy()
    detached_bm.faces.ensure_lookup_table()
    faces_to_drop = [f for f in detached_bm.faces if f.index not in iso_indices]
    if faces_to_drop:
        bmesh.ops.delete(detached_bm, geom=faces_to_drop, context="FACES")
    orphan_det = [v for v in detached_bm.verts if not v.link_faces]
    if orphan_det:
        bmesh.ops.delete(detached_bm, geom=orphan_det, context="VERTS")
    if len(detached_bm.faces) > 0:
        bmesh.ops.recalc_face_normals(detached_bm, faces=detached_bm.faces[:])

    detached_obj = _create_chunk_object(
        detached_bm,
        host,
        name=part_name,
        center_pivot=True,
    )
    detached_bm.free()

    host_seam_verts = {bm.verts[i] for i in seam_vert_indices if i < len(bm.verts)}
    bmesh.ops.delete(bm, geom=isolated_faces, context="FACES")
    orphan_host = [v for v in bm.verts if not v.link_faces]
    if orphan_host:
        bmesh.ops.delete(bm, geom=orphan_host, context="VERTS")

    if cap_wound:
        wound_edges = [
            e for e in bm.edges
            if e.is_valid and e.is_boundary and (e.verts[0] in host_seam_verts and e.verts[1] in host_seam_verts)
        ]
        cap_idx = None
        if cap_material_name:
            cap_matches = _find_material_indices(host, cap_material_name)
            if cap_matches:
                cap_idx = cap_matches[0]
            else:
                mat = bpy.data.materials.get(cap_material_name) or make_pbr_material(name=cap_material_name)
                host.data.materials.append(mat)
                cap_idx = len(host.data.materials) - 1
        _fill_boundary_loops(
            bm,
            wound_edges,
            interior_mat_idx=cap_idx,
            expected_outward_no=(local_no if (has_plane and side_pos) else (-local_no if has_plane else None)),
            smooth_cap=False,
        )

    if len(bm.faces) > 0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(host.data)
    bm.free()
    host.data.update()

    if shell_thickness is not None and float(shell_thickness) > 0.0:
        mod = detached_obj.modifiers.new(name="Solidify_Shell", type="SOLIDIFY")
        mod.thickness = float(shell_thickness)
        mod.offset = float(solidify_offset)
        mod.use_rim = True
        mod.use_even_offset = True
        if hasattr(mod, "use_quality_normals"):
            mod.use_quality_normals = True
        if apply_solidify:
            sanitize_for_surgery(detached_obj, merge_dist=1e-6)

    return detached_obj


def split_or_fracture(
    obj,
    num_shards=6,
    seed=42,
    interior_material=None,
    interior_material_name="Fracture_Interior",
    prefix=None,
    keep_original=False,
    min_island_faces=4,
):
    """
    Universal Op 2: Splits loose disconnected islands automatically when
    `loose_island_count > 1` (e.g., Onix boulders), or performs deterministic
    headless-safe `bmesh.ops.bisect_plane` shattering with an interior material
    applied strictly to newly created cut cap faces.
    """
    src = _resolve_mesh_object(obj)
    if len(src.modifiers) > 0:
        sanitize_for_surgery(src)

    base_name = prefix or f"{src.name}_Part"
    src_colls = list(src.users_collection)
    src_mats = [s.material for s in src.material_slots]

    bm = bmesh.new()
    bm.from_mesh(src.data)
    if len(bm.faces) == 0:
        bm.free()
        raise ValueError(f"Object '{src.name}' has no faces to split or fracture.")

    bm.verts.index_update()
    bm.faces.index_update()
    islands = _find_bm_face_islands(bm)
    substantial = [isl for isl in islands if len(isl) >= int(min_island_faces)]

    if len(substantial) > 1 or (len(islands) > 1 and len(substantial) == len(islands)):
        target_islands = substantial if len(substantial) > 1 else islands

        def _isl_key(isl):
            verts = {v for f in isl for v in f.verts}
            c = sum((v.co for v in verts), mathutils.Vector((0.0, 0.0, 0.0))) / max(len(verts), 1)
            return (round(c.z, 4), round(c.y, 4), round(c.x, 4))

        target_islands.sort(key=_isl_key)
        created = []
        for idx, isl in enumerate(target_islands):
            iso_idx = {f.index for f in isl}
            sub_bm = bm.copy()
            sub_bm.faces.ensure_lookup_table()
            drop = [f for f in sub_bm.faces if f.index not in iso_idx]
            if drop:
                bmesh.ops.delete(sub_bm, geom=drop, context="FACES")
            orphans = [v for v in sub_bm.verts if not v.link_faces]
            if orphans:
                bmesh.ops.delete(sub_bm, geom=orphans, context="VERTS")
            if len(sub_bm.faces) > 0:
                bmesh.ops.recalc_face_normals(sub_bm, faces=sub_bm.faces[:])
            shard_obj = _create_chunk_object(
                sub_bm,
                src,
                name=f"{base_name}_{idx:02d}",
                src_materials=src_mats,
                center_pivot=True,
                collections=src_colls,
            )
            sub_bm.free()
            created.append(shard_obj)

        bm.free()
        if not keep_original:
            bpy.data.objects.remove(src, do_unlink=True)
        return created

    target_count = max(2, int(num_shards))
    interior_mat, interior_idx = _ensure_interior_material(
        src_mats,
        interior_material=interior_material,
        interior_material_name=interior_material_name,
    )
    if interior_idx >= len(src_mats):
        src_mats.append(interior_mat)

    rand = _make_lcg(seed)
    chunks = [bm]
    max_attempts = target_count * 10
    attempts = 0

    while len(chunks) < target_count and attempts < max_attempts:
        attempts += 1

        def _chunk_score(c_bm):
            if not c_bm.verts:
                return 0.0
            xs = [v.co.x for v in c_bm.verts]
            ys = [v.co.y for v in c_bm.verts]
            zs = [v.co.z for v in c_bm.verts]
            return (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs)) + 1e-6 * len(c_bm.faces)

        chunks.sort(key=_chunk_score, reverse=True)
        cur_bm = chunks[0]
        if len(cur_bm.faces) < 8:
            break

        xs = [v.co.x for v in cur_bm.verts]
        ys = [v.co.y for v in cur_bm.verts]
        zs = [v.co.z for v in cur_bm.verts]
        min_v = mathutils.Vector((min(xs), min(ys), min(zs)))
        max_v = mathutils.Vector((max(xs), max(ys), max(zs)))
        ext = max_v - min_v
        center = 0.5 * (min_v + max_v)

        dominant = [0.0, 0.0, 0.0]
        max_axis = max(range(3), key=lambda i: ext[i])
        dominant[max_axis] = 1.0
        jitter = mathutils.Vector((rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0)) * 0.65
        plane_no = mathutils.Vector(dominant) + jitter
        if plane_no.length < 1e-5:
            plane_no = mathutils.Vector((0.0, 0.0, 1.0))
        plane_no.normalize()

        span_along_no = abs(ext.x * plane_no.x) + abs(ext.y * plane_no.y) + abs(ext.z * plane_no.z)
        plane_co = center + plane_no * ((rand() - 0.5) * 0.30 * span_along_no)

        cur_bm.verts.index_update()
        cur_bm.faces.index_update()
        bm_pos = cur_bm.copy()
        bm_neg = cur_bm.copy()

        ret_pos = bmesh.ops.bisect_plane(
            bm_pos,
            geom=bm_pos.verts[:] + bm_pos.edges[:] + bm_pos.faces[:],
            plane_co=plane_co,
            plane_no=plane_no,
            dist=1e-5,
            clear_inner=True,
            clear_outer=False,
        )
        ret_neg = bmesh.ops.bisect_plane(
            bm_neg,
            geom=bm_neg.verts[:] + bm_neg.edges[:] + bm_neg.faces[:],
            plane_co=plane_co,
            plane_no=plane_no,
            dist=1e-5,
            clear_inner=False,
            clear_outer=True,
        )

        for half_bm in (bm_pos, bm_neg):
            orphans = [v for v in half_bm.verts if not v.link_faces]
            if orphans:
                bmesh.ops.delete(half_bm, geom=orphans, context="VERTS")

        if len(bm_pos.faces) < 4 or len(bm_neg.faces) < 4:
            bm_pos.free()
            bm_neg.free()
            continue

        for half_bm, ret, outward_no in (
            (bm_pos, ret_pos, -plane_no),
            (bm_neg, ret_neg, plane_no),
        ):
            cut_set = {
                g for g in ret.get("geom_cut", [])
                if isinstance(g, bmesh.types.BMEdge) and g.is_valid and g.is_boundary
            }
            for e in half_bm.edges:
                if e.is_valid and e.is_boundary:
                    if (
                        abs((e.verts[0].co - plane_co).dot(plane_no)) < 1e-4
                        and abs((e.verts[1].co - plane_co).dot(plane_no)) < 1e-4
                    ):
                        cut_set.add(e)
            _fill_boundary_loops(
                half_bm,
                list(cut_set),
                interior_mat_idx=interior_idx,
                expected_outward_no=outward_no,
                smooth_cap=False,
            )

        chunks.pop(0)
        cur_bm.free()
        chunks.append(bm_pos)
        chunks.append(bm_neg)

    created_shards = []
    for idx, c_bm in enumerate(chunks):
        shard_obj = _create_chunk_object(
            c_bm,
            src,
            name=f"{base_name}_{idx:02d}",
            src_materials=src_mats,
            center_pivot=True,
            collections=src_colls,
        )
        c_bm.free()
        created_shards.append(shard_obj)

    if not keep_original:
        bpy.data.objects.remove(src, do_unlink=True)

    return created_shards


def extract_prop(
    host_obj,
    material_name=None,
    replacement_prop=None,
    replacement_blend_path=None,
    replacement_names=None,
    close_pocket=True,
    pocket_material_name=None,
    scale_mode="MAX",
    match_parent=True,
    auto_detect_open_shell=True,
    search_bounds_center=None,
    search_radius=None,
    fallback_shape="EGG",
    prop_name="Extracted_Prop",
):
    """
    Universal Op 3: Deletes clipped partial faces (`is_closed_shell == False`,
    e.g., Chansey's pouch egg), closes the host pocket, and snaps a complete
    watertight replacement prop into the exact local center and size.
    """
    host = _resolve_mesh_object(host_obj)
    if len(host.modifiers) > 0:
        sanitize_for_surgery(host)

    bm = bmesh.new()
    bm.from_mesh(host.data)

    target_faces = []
    if material_name is not None:
        mat_indices = set(_find_material_indices(host, material_name))
        if not mat_indices:
            bm.free()
            avail = [s.material.name for s in host.material_slots if s.material]
            raise ValueError(f"Material '{material_name}' not found on '{host.name}'. Available: {avail}")
        for f in bm.faces:
            if f.material_index in mat_indices:
                if search_bounds_center is not None and search_radius is not None:
                    w_center = host.matrix_world @ f.calc_center_median()
                    if (w_center - mathutils.Vector(search_bounds_center)).length > float(search_radius):
                        continue
                target_faces.append(f)
    elif auto_detect_open_shell:
        islands = _find_bm_face_islands(bm)
        open_islands = [isl for isl in islands if not _is_face_subset_closed(isl)]
        if not open_islands:
            bm.free()
            raise ValueError(f"No open partial shells (is_closed_shell == False) found on '{host.name}'.")
        if search_bounds_center is not None:
            ref_pt = mathutils.Vector(search_bounds_center)

            def _dist(isl):
                verts = {v for f in isl for v in f.verts}
                wc = host.matrix_world @ (sum((v.co for v in verts), mathutils.Vector()) / max(len(verts), 1))
                return (wc - ref_pt).length

            open_islands.sort(key=_dist)
            target_faces = open_islands[0]
        else:
            open_islands.sort(key=len)
            target_faces = open_islands[0]
    else:
        bm.free()
        raise ValueError("extract_prop requires `material_name` or `auto_detect_open_shell=True`.")

    if not target_faces:
        bm.free()
        raise ValueError(f"No matching partial prop faces found on '{host.name}'.")

    was_closed_shell = _is_face_subset_closed(target_faces)

    unique_verts = {v for f in target_faces for v in f.verts}
    world_pts = [host.matrix_world @ v.co for v in unique_verts]
    inf = float("inf")
    clip_min = mathutils.Vector((inf, inf, inf))
    clip_max = mathutils.Vector((-inf, -inf, -inf))
    for p in world_pts:
        clip_min.x = min(clip_min.x, p.x)
        clip_min.y = min(clip_min.y, p.y)
        clip_min.z = min(clip_min.z, p.z)
        clip_max.x = max(clip_max.x, p.x)
        clip_max.y = max(clip_max.y, p.y)
        clip_max.z = max(clip_max.z, p.z)

    clip_center_world = 0.5 * (clip_min + clip_max)
    clip_size_world = clip_max - clip_min

    mat_counts = {}
    for f in target_faces:
        mat_counts[f.material_index] = mat_counts.get(f.material_index, 0) + 1
    dom_idx = max(mat_counts, key=mat_counts.get)
    dom_mat = host.material_slots[dom_idx].material if dom_idx < len(host.material_slots) else None

    target_set = set(target_faces)
    bm.verts.ensure_lookup_table()
    seam_vert_indices = set()
    for e in bm.edges:
        has_t = any(f in target_set for f in e.link_faces)
        has_h = any(f not in target_set for f in e.link_faces)
        if has_t and has_h:
            seam_vert_indices.add(e.verts[0].index)
            seam_vert_indices.add(e.verts[1].index)

    host_seam_verts = {bm.verts[i] for i in seam_vert_indices if i < len(bm.verts)}

    bmesh.ops.delete(bm, geom=target_faces, context="FACES")
    orphan_verts = [v for v in bm.verts if not v.link_faces]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context="VERTS")

    if close_pocket:
        pocket_edges = [
            e for e in bm.edges
            if e.is_valid and e.is_boundary and (e.verts[0] in host_seam_verts and e.verts[1] in host_seam_verts)
        ]
        if not pocket_edges:
            max_search = max(clip_size_world.length * 1.5, 0.12)
            pocket_edges = [
                e for e in bm.edges
                if e.is_valid and e.is_boundary
                and ((host.matrix_world @ (0.5 * (e.verts[0].co + e.verts[1].co))) - clip_center_world).length <= max_search
            ]

        pocket_mat_idx = None
        if pocket_material_name:
            matches = _find_material_indices(host, pocket_material_name)
            if matches:
                pocket_mat_idx = matches[0]
            else:
                pmat = bpy.data.materials.get(pocket_material_name) or make_pbr_material(name=pocket_material_name)
                host.data.materials.append(pmat)
                pocket_mat_idx = len(host.data.materials) - 1

        if pocket_edges:
            _fill_boundary_loops(bm, pocket_edges, interior_mat_idx=pocket_mat_idx, smooth_cap=True)

    if len(bm.faces) > 0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(host.data)
    bm.free()
    host.data.update()

    if replacement_prop is None and replacement_blend_path is not None:
        replacement_prop = append_from_blend(
            replacement_blend_path,
            names=replacement_names,
            parent_name=prop_name,
            target_height_m=None,
            ground_to_z=None,
        )
    elif replacement_prop is None:
        rep_bm = bmesh.new()
        shape = str(fallback_shape).strip().upper()
        if shape == "CUBE":
            bmesh.ops.create_cube(rep_bm, size=1.0)
        else:
            bmesh.ops.create_uvsphere(rep_bm, u_segments=28, v_segments=18, radius=0.5)
            if shape == "EGG":
                for v in rep_bm.verts:
                    t = v.co.z + 0.5
                    taper = 1.08 - 0.30 * (t ** 1.4)
                    v.co.x *= taper
                    v.co.y *= taper
                    v.co.z *= 1.28
        for f in rep_bm.faces:
            f.smooth = True
        bmesh.ops.recalc_face_normals(rep_bm, faces=rep_bm.faces[:])
        rep_mesh = bpy.data.meshes.new(prop_name + "_Mesh")
        rep_bm.to_mesh(rep_mesh)
        rep_bm.free()
        if dom_mat is not None:
            rep_mesh.materials.append(dom_mat)
        replacement_prop = bpy.data.objects.new(prop_name, rep_mesh)
        _link_object_to_collections(replacement_prop, host.users_collection)

    bpy.context.view_layer.update()
    rep_min, rep_max = get_world_bounds(replacement_prop)
    rep_size = rep_max - rep_min

    mode = str(scale_mode).strip().upper()
    if mode == "MAX":
        target_d = max(clip_size_world.x, clip_size_world.y, clip_size_world.z)
        rep_d = max(rep_size.x, rep_size.y, rep_size.z, 1e-6)
        replacement_prop.scale = replacement_prop.scale * (target_d / rep_d)
    elif mode == "EXACT":
        replacement_prop.scale.x *= clip_size_world.x / max(rep_size.x, 1e-6)
        replacement_prop.scale.y *= clip_size_world.y / max(rep_size.y, 1e-6)
        replacement_prop.scale.z *= clip_size_world.z / max(rep_size.z, 1e-6)

    bpy.context.view_layer.update()
    rep_min, rep_max = get_world_bounds(replacement_prop)
    rep_center = 0.5 * (rep_min + rep_max)
    delta = clip_center_world - rep_center

    mat_w = replacement_prop.matrix_world.copy()
    mat_w.translation = mat_w.translation + delta
    replacement_prop.matrix_world = mat_w
    bpy.context.view_layer.update()

    if match_parent and host.parent is not None and replacement_prop.parent is None:
        saved_world = mat_w.copy()
        replacement_prop.parent = host.parent
        replacement_prop.matrix_parent_inverse = host.parent.matrix_world.inverted()
        replacement_prop.matrix_world = saved_world
        bpy.context.view_layer.update()

    replacement_prop["is_closed_shell_source"] = bool(was_closed_shell)
    return replacement_prop


def slice_and_carve(
    obj,
    plane_co=(0.0, 0.0, 0.0),
    plane_no=(0.0, 0.0, 1.0),
    axis=None,
    threshold=None,
    coord_space="WORLD",
    interior_material=None,
    interior_material_name="Cross_Section_Interior",
    name_a=None,
    name_b=None,
    keep_original=False,
    center_pivots=False,
):
    """
    Universal Op 4: Bisects `obj` cleanly along any plane into two capped
    watertight halves `(half_a, half_b)` with an interior cross-section material
    applied strictly to the newly created cut caps.
    """
    src = _resolve_mesh_object(obj)
    if len(src.modifiers) > 0:
        sanitize_for_surgery(src)

    co, no = _plane_from_args(plane_co=plane_co, plane_no=plane_no, axis=axis, threshold=threshold)
    if str(coord_space).upper() == "WORLD":
        local_co, local_no = _world_plane_to_local(src, co, no)
    else:
        local_co, local_no = co, no

    src_colls = list(src.users_collection)
    src_mats = [s.material for s in src.material_slots]
    interior_mat, interior_idx = _ensure_interior_material(
        src_mats,
        interior_material=interior_material,
        interior_material_name=interior_material_name,
    )
    if interior_idx >= len(src_mats):
        src_mats.append(interior_mat)

    bm_base = bmesh.new()
    bm_base.from_mesh(src.data)
    bm_base.verts.index_update()
    bm_base.faces.index_update()
    bm_a = bm_base.copy()
    bm_b = bm_base.copy()
    bm_base.free()

    ret_a = bmesh.ops.bisect_plane(
        bm_a,
        geom=bm_a.verts[:] + bm_a.edges[:] + bm_a.faces[:],
        plane_co=local_co,
        plane_no=local_no,
        dist=1e-5,
        clear_inner=True,
        clear_outer=False,
    )
    ret_b = bmesh.ops.bisect_plane(
        bm_b,
        geom=bm_b.verts[:] + bm_b.edges[:] + bm_b.faces[:],
        plane_co=local_co,
        plane_no=local_no,
        dist=1e-5,
        clear_inner=False,
        clear_outer=True,
    )

    for half_bm in (bm_a, bm_b):
        orphans = [v for v in half_bm.verts if not v.link_faces]
        if orphans:
            bmesh.ops.delete(half_bm, geom=orphans, context="VERTS")

    if len(bm_a.faces) == 0 or len(bm_b.faces) == 0:
        bm_a.free()
        bm_b.free()
        raise ValueError(
            f"Plane ({co}, {no}) did not intersect '{src.name}' cleanly (one half was empty)."
        )

    for half_bm, ret, outward_no in (
        (bm_a, ret_a, -local_no),
        (bm_b, ret_b, local_no),
    ):
        cut_edges = {
            g for g in ret.get("geom_cut", [])
            if isinstance(g, bmesh.types.BMEdge) and g.is_valid and g.is_boundary
        }
        for e in half_bm.edges:
            if e.is_valid and e.is_boundary:
                if (
                    abs((e.verts[0].co - local_co).dot(local_no)) < 1e-4
                    and abs((e.verts[1].co - local_co).dot(local_no)) < 1e-4
                ):
                    cut_edges.add(e)
        _fill_boundary_loops(
            half_bm,
            list(cut_edges),
            interior_mat_idx=interior_idx,
            expected_outward_no=outward_no,
            smooth_cap=False,
        )

    half_a = _create_chunk_object(
        bm_a,
        src,
        name=name_a or f"{src.name}_HalfA",
        src_materials=src_mats,
        center_pivot=center_pivots,
        collections=src_colls,
    )
    half_b = _create_chunk_object(
        bm_b,
        src,
        name=name_b or f"{src.name}_HalfB",
        src_materials=src_mats,
        center_pivot=center_pivots,
        collections=src_colls,
    )

    bm_a.free()
    bm_b.free()

    if not keep_original:
        bpy.data.objects.remove(src, do_unlink=True)

    return half_a, half_b


def deform_and_morph(
    obj,
    shape_names=None,
    deformations=None,
    amplitude=0.18,
    axis="Z",
    world_aligned=True,
):
    """
    Universal Op 5: Safely authors named Shape Keys (`Basis`, `Risen`, `Jiggle`,
    `Melted`, `Squashed`, `Inflated`, or custom functions/vertex targets).

    When `world_aligned=True` (default), presets evaluate height and radial
    spread in world-aligned space (`+Z` up) before writing back into local vertex
    space—preventing unbaked glTF `+90° X` meshes from puffing sideways.
    """
    mesh_obj = _resolve_mesh_object(obj)
    mesh = mesh_obj.data
    vert_count = len(mesh.vertices)
    if vert_count == 0:
        raise ValueError(f"Mesh '{mesh_obj.name}' has 0 vertices.")

    if mesh.shape_keys is None:
        mesh_obj.shape_key_add(name="Basis", from_mix=False)

    basis_kb = mesh.shape_keys.key_blocks["Basis"]
    basis_flat = [0.0] * (vert_count * 3)
    basis_kb.data.foreach_get("co", basis_flat)

    local_basis = [
        mathutils.Vector((basis_flat[i * 3], basis_flat[i * 3 + 1], basis_flat[i * 3 + 2]))
        for i in range(vert_count)
    ]
    local_normals = [v.normal.copy() for v in mesh.vertices]

    bpy.context.view_layer.update()
    if world_aligned:
        rot_3 = mesh_obj.matrix_world.to_3x3()
        inv_rot_3 = rot_3.inverted()
        eval_vecs = [rot_3 @ co for co in local_basis]
        eval_normals = [(rot_3 @ n).normalized() if n.length > 1e-8 else n for n in local_normals]
    else:
        inv_rot_3 = mathutils.Matrix.Identity(3)
        eval_vecs = local_basis
        eval_normals = local_normals

    inf = float("inf")
    b_min = mathutils.Vector((inf, inf, inf))
    b_max = mathutils.Vector((-inf, -inf, -inf))
    for co in eval_vecs:
        b_min.x = min(b_min.x, co.x)
        b_min.y = min(b_min.y, co.y)
        b_min.z = min(b_min.z, co.z)
        b_max.x = max(b_max.x, co.x)
        b_max.y = max(b_max.y, co.y)
        b_max.z = max(b_max.z, co.z)

    size = b_max - b_min
    center = 0.5 * (b_min + b_max)
    safe_sz = mathutils.Vector((max(size.x, 1e-5), max(size.y, 1e-5), max(size.z, 1e-5)))

    if shape_names is None and deformations is None:
        names_to_author = ["Risen", "Jiggle", "Melted"]
    elif shape_names is None:
        names_to_author = list(deformations.keys())
    elif isinstance(shape_names, str):
        names_to_author = [shape_names]
    else:
        names_to_author = list(shape_names)

    if deformations:
        for k in deformations:
            if k not in names_to_author:
                names_to_author.append(k)

    def _compute_preset(preset_name, amp):
        pname = str(preset_name).strip().upper()
        out = [0.0] * (vert_count * 3)
        rx = max(0.5 * safe_sz.x, 1e-5)
        ry = max(0.5 * safe_sz.y, 1e-5)

        for i, co in enumerate(eval_vecs):
            h = max(0.0, min(1.0, (co.z - b_min.z) / safe_sz.z))
            dx = co.x - center.x
            dy = co.y - center.y
            r_norm = min(1.5, math.hypot(dx / rx, dy / ry))
            nx, ny, nz = co.x, co.y, co.z

            if pname == "RISEN":
                dome = math.sin(h * math.pi * 0.5) ** 0.85
                puff = 1.0 + amp * 1.15 * (h * (1.0 - 0.25 * min(r_norm, 1.0)))
                nx = center.x + dx * puff
                ny = center.y + dy * puff
                nz = (
                    b_min.z
                    + (co.z - b_min.z) * (1.0 + amp * 2.2 * dome)
                    + amp * 0.35 * safe_sz.z * dome * max(0.0, 1.0 - 0.55 * (r_norm ** 2))
                )
            elif pname == "JIGGLE":
                wave_x = math.sin(h * math.pi * 0.90)
                wave_y = math.sin(h * math.pi * 1.50)
                nx = co.x + amp * 0.65 * safe_sz.x * wave_x
                ny = co.y + amp * 0.30 * safe_sz.y * wave_y
                nz = b_min.z + (co.z - b_min.z) * (1.0 - amp * 0.25 * h)
            elif pname == "MELTED":
                slump = max(0.15, 1.0 - min(0.82, amp * 3.0))
                spread = 1.0 + amp * 2.6 * (1.0 - 0.45 * h)
                nx = center.x + dx * spread
                ny = center.y + dy * spread
                nz = b_min.z + (co.z - b_min.z) * slump
            elif pname == "SQUASHED":
                comp = max(0.25, 1.0 - amp * 1.8)
                bulge = 1.0 / math.sqrt(comp)
                ax = str(axis).strip().upper()
                if ax == "X":
                    nx = center.x + dx * comp
                    ny = center.y + dy * bulge
                    nz = center.z + (co.z - center.z) * bulge
                elif ax == "Y":
                    nx = center.x + dx * bulge
                    ny = center.y + dy * comp
                    nz = center.z + (co.z - center.z) * bulge
                else:
                    nx = center.x + dx * bulge
                    ny = center.y + dy * bulge
                    nz = b_min.z + (co.z - b_min.z) * comp
            elif pname == "INFLATED":
                n_vec = eval_normals[i]
                shift = amp * min(safe_sz.x, safe_sz.y, safe_sz.z) * 0.5
                nx = co.x + n_vec.x * shift
                ny = co.y + n_vec.y * shift
                nz = co.z + n_vec.z * shift

            deformed_local = inv_rot_3 @ mathutils.Vector((nx, ny, nz)) if world_aligned else mathutils.Vector((nx, ny, nz))
            out[i * 3] = float(deformed_local.x)
            out[i * 3 + 1] = float(deformed_local.y)
            out[i * 3 + 2] = float(deformed_local.z)
        return out

    deform_map = deformations or {}
    for s_name in names_to_author:
        if s_name == "Basis":
            continue
        kb = mesh.shape_keys.key_blocks.get(s_name)
        if kb is None:
            kb = mesh_obj.shape_key_add(name=s_name, from_mix=False)

        spec = deform_map.get(s_name)
        if callable(spec):
            flat = [0.0] * (vert_count * 3)
            for i, co in enumerate(local_basis):
                rel = mathutils.Vector((
                    (eval_vecs[i].x - b_min.x) / safe_sz.x,
                    (eval_vecs[i].y - b_min.y) / safe_sz.y,
                    (eval_vecs[i].z - b_min.z) / safe_sz.z,
                ))
                try:
                    res = spec(co.copy(), local_normals[i], rel, b_min, b_max)
                except TypeError:
                    res = spec(co.copy())
                flat[i * 3] = float(res[0])
                flat[i * 3 + 1] = float(res[1])
                flat[i * 3 + 2] = float(res[2])
        elif isinstance(spec, bpy.types.Object) and spec.type == "MESH":
            if len(spec.data.vertices) != vert_count:
                raise ValueError(
                    f"Target mesh '{spec.name}' has {len(spec.data.vertices)} verts; "
                    f"expected {vert_count} to match '{mesh_obj.name}'."
                )
            flat = [0.0] * (vert_count * 3)
            spec.data.vertices.foreach_get("co", flat)
        elif isinstance(spec, (list, tuple)):
            if len(spec) != vert_count:
                raise ValueError(f"Expected {vert_count} vertex positions for '{s_name}', got {len(spec)}.")
            flat = []
            for pt in spec:
                flat.extend((float(pt[0]), float(pt[1]), float(pt[2])))
        elif isinstance(spec, dict):
            flat = _compute_preset(spec.get("preset", s_name), float(spec.get("amplitude", amplitude)))
        else:
            flat = _compute_preset(s_name, float(amplitude))

        kb.data.foreach_set("co", flat)
        kb.value = 0.0
        kb.slider_min = 0.0
        kb.slider_max = 1.0

    mesh.update()
    return mesh_obj


def contain_and_emit(
    cavity_bounds=None,
    container_obj=None,
    name="Liquid_Volume",
    fill_ratio=0.65,
    meniscus_height=0.004,
    subdivisions=32,
    cap_rings=5,
    surface_ripple=0.0,
    material=None,
    wall_clearance=0.002,
    parent=None,
    add_emitter=False,
    particle_count=180,
):
    """
    Universal Op 6: Builds a watertight liquid/batter volume mesh mathematically
    constrained inside `cavity_bounds` (`inner_radius`, `top_radius`, `floor_z`
    to `rim_z`, `cavity_center_xy`), with optional surface particle emitter.
    """
    if isinstance(cavity_bounds, str):
        if os.path.isfile(cavity_bounds):
            with open(cavity_bounds, "r", encoding="utf-8") as fh:
                cavity_bounds = json.load(fh)
        else:
            cavity_bounds = json.loads(cavity_bounds)

    if cavity_bounds is None:
        if container_obj is None:
            raise ValueError("contain_and_emit requires `cavity_bounds` or `container_obj`.")
        c_min, c_max = get_world_bounds(container_obj)
        c_sz = c_max - c_min
        bounds = {
            "cavity_center_xy": (0.5 * (c_min.x + c_max.x), 0.5 * (c_min.y + c_max.y)),
            "inner_radius": max(0.01, 0.5 * min(c_sz.x, c_sz.y) * 0.82),
            "top_radius": max(0.01, 0.5 * min(c_sz.x, c_sz.y) * 0.86),
            "floor_z": c_min.z + c_sz.z * 0.08,
            "rim_z": c_max.z - c_sz.z * 0.05,
        }
    else:
        bounds = dict(cavity_bounds)

    if "cavity_center_xy" in bounds:
        cx, cy = float(bounds["cavity_center_xy"][0]), float(bounds["cavity_center_xy"][1])
    elif "center_xy" in bounds:
        cx, cy = float(bounds["center_xy"][0]), float(bounds["center_xy"][1])
    elif "center" in bounds:
        cx, cy = float(bounds["center"][0]), float(bounds["center"][1])
    else:
        cx = float(bounds.get("cx", 0.0))
        cy = float(bounds.get("cy", 0.0))

    floor_z = float(bounds.get("floor_z", bounds.get("min_z", bounds.get("bottom_z", 0.0))))
    rim_z = float(bounds.get("rim_z", bounds.get("max_z", bounds.get("top_z", floor_z + 0.1))))
    if rim_z <= floor_z:
        raise ValueError(f"Invalid cavity_bounds: rim_z ({rim_z}) must exceed floor_z ({floor_z}).")

    clamped_fill = max(0.02, min(0.98, float(fill_ratio)))
    clear = max(0.0, float(wall_clearance))
    z_bot = min(floor_z + clear, rim_z - 1e-4)
    z_top = min(rim_z - clear, floor_z + (rim_z - floor_z) * clamped_fill)
    if z_top <= z_bot:
        z_top = z_bot + max(1e-4, (rim_z - floor_z) * 0.1)

    half_ext = bounds.get("half_extents_xy")
    bm = bmesh.new()

    if half_ext is not None:
        hx = max(1e-4, float(half_ext[0]) - clear)
        hy = max(1e-4, float(half_ext[1]) - clear)
        hz = 0.5 * (z_top - z_bot)
        zc = 0.5 * (z_bot + z_top)
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x = cx + v.co.x * (2.0 * hx)
            v.co.y = cy + v.co.y * (2.0 * hy)
            v.co.z = zc + v.co.z * (2.0 * hz)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=max(2, int(cap_rings)), use_grid_fill=True)
        for v in bm.verts:
            if abs(v.co.z - z_top) < 1e-5:
                edge_dist = max(abs(v.co.x - cx) / hx, abs(v.co.y - cy) / hy)
                v.co.z += float(meniscus_height) * (edge_dist ** 3)
                if surface_ripple > 0.0:
                    r = math.hypot(v.co.x - cx, v.co.y - cy)
                    v.co.z += float(surface_ripple) * math.cos(r * 45.0) * (1.0 - edge_dist * 0.5)
    else:
        r_bot_raw = float(bounds.get("inner_radius_bottom") or bounds.get("inner_radius") or bounds.get("radius") or 0.1)
        r_top_raw = float(bounds.get("inner_radius_rim") or bounds.get("top_radius") or bounds.get("inner_radius") or r_bot_raw)
        t_bot = (z_bot - floor_z) / (rim_z - floor_z)
        t_top = (z_top - floor_z) / (rim_z - floor_z)
        r_bot = max(1e-4, (r_bot_raw + (r_top_raw - r_bot_raw) * t_bot) - clear)
        r_top = max(1e-4, (r_bot_raw + (r_top_raw - r_bot_raw) * t_top) - clear)

        segs = max(12, int(subdivisions))
        v_steps = 5
        rings_c = max(2, int(cap_rings))
        men = min(float(meniscus_height), max(0.0, (rim_z - z_top) * 0.5))

        v_bot_center = bm.verts.new((cx, cy, z_bot))
        side_rings = []
        for vz in range(v_steps + 1):
            t = vz / v_steps
            z_cur = z_bot + (z_top - z_bot) * t
            r_cur = r_bot + (r_top - r_bot) * t
            ring = []
            for s in range(segs):
                ang = (2.0 * math.pi * s) / segs
                ring.append(bm.verts.new((cx + r_cur * math.cos(ang), cy + r_cur * math.sin(ang), z_cur)))
            side_rings.append(ring)

        for s in range(segs):
            s_next = (s + 1) % segs
            bm.faces.new((v_bot_center, side_rings[0][s_next], side_rings[0][s]))

        for vz in range(v_steps):
            r0 = side_rings[vz]
            r1 = side_rings[vz + 1]
            for s in range(segs):
                s_next = (s + 1) % segs
                bm.faces.new((r0[s], r0[s_next], r1[s_next], r1[s]))

        top_rings = []
        for cr in range(1, rings_c):
            frac = 1.0 - (cr / rings_c)
            rr = r_top * frac
            z_off = men * (frac ** 3)
            if surface_ripple > 0.0:
                z_off += float(surface_ripple) * math.cos(frac * math.pi * 3.0)
            ring = []
            for s in range(segs):
                ang = (2.0 * math.pi * s) / segs
                ring.append(bm.verts.new((cx + rr * math.cos(ang), cy + rr * math.sin(ang), z_top + z_off)))
            top_rings.append(ring)

        outer_ring = side_rings[-1]
        for v in outer_ring:
            v.co.z = min(rim_z - clear * 0.5, z_top + men)

        prev_r = outer_ring
        for inner_r in top_rings:
            for s in range(segs):
                s_next = (s + 1) % segs
                bm.faces.new((prev_r[s], prev_r[s_next], inner_r[s_next], inner_r[s]))
            prev_r = inner_r

        z_center_top = z_top + (float(surface_ripple) if surface_ripple > 0.0 else 0.0)
        v_top_center = bm.verts.new((cx, cy, z_center_top))
        for s in range(segs):
            s_next = (s + 1) % segs
            bm.faces.new((prev_r[s], prev_r[s_next], v_top_center))

    for f in bm.faces:
        f.smooth = True
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    if material is not None:
        mesh.materials.append(material)

    liq_obj = bpy.data.objects.new(name, mesh)
    if container_obj is not None and len(container_obj.users_collection) > 0:
        _link_object_to_collections(liq_obj, container_obj.users_collection)
    else:
        _link_object_to_collections(liq_obj, None)

    if parent is True and container_obj is not None:
        liq_obj.parent = container_obj
        liq_obj.matrix_parent_inverse = container_obj.matrix_world.inverted()
    elif isinstance(parent, bpy.types.Object):
        liq_obj.parent = parent
        liq_obj.matrix_parent_inverse = parent.matrix_world.inverted()

    if add_emitter:
        em_bm = bmesh.new()
        r_em = float(bounds.get("top_radius", bounds.get("inner_radius", 0.1))) * 0.75
        bmesh.ops.create_circle(em_bm, cap_ends=True, radius=max(1e-4, r_em), segments=20)
        for v in em_bm.verts:
            v.co.x += cx
            v.co.y += cy
            v.co.z = max(z_bot, z_top - 0.002)
        em_mesh = bpy.data.meshes.new(name + "_EmitterMesh")
        em_bm.to_mesh(em_mesh)
        em_bm.free()
        em_obj = bpy.data.objects.new(name + "_Emitter", em_mesh)
        _link_object_to_collections(em_obj, liq_obj.users_collection)
        em_obj.parent = liq_obj
        em_obj.hide_render = True
        p_mod = em_obj.modifiers.new(name="Liquid_Particles", type="PARTICLE_SYSTEM")
        pset = p_mod.particle_system.settings
        pset.count = int(particle_count)
        pset.frame_start = 1
        pset.frame_end = 120
        pset.lifetime = 35
        pset.emit_from = "FACE"
        pset.physics_type = "NEWTON"
        pset.normal_factor = 0.25
        pset.particle_size = max(0.002, r_em * 0.08)

    liq_obj["cavity_bounds_json"] = json.dumps(bounds)
    return liq_obj


# ==============================================================================
# MODULE 3 — BLENDER 4.5 PROCEDURAL & X-RAY SHADERS
# ==============================================================================

def make_pbr_material(
    name="Procedural_PBR",
    base_color=(0.8, 0.8, 0.8, 1.0),
    secondary_color=None,
    metallic=0.0,
    roughness=0.45,
    roughness_variation=0.12,
    pattern="NOISE",
    pattern_scale=12.0,
    pattern_detail=6.0,
    bump_strength=0.08,
    subsurface_weight=0.0,
    subsurface_scale=0.03,
    subsurface_radius=(1.0, 0.35, 0.2),
    transmission_weight=0.0,
    ior=1.45,
    coat_weight=0.0,
    coat_roughness=0.1,
    specular_ior_level=0.5,
    emission_color=(0.0, 0.0, 0.0, 1.0),
    emission_strength=0.0,
    alpha=1.0,
    coord_mode="OBJECT",
    overwrite=True,
):
    """
    Build a procedural Noise/Voronoi/ColorRamp material wired to verified
    Blender 4.5 Principled BSDF socket names (`Subsurface Weight`,
    `Specular IOR Level`, `Transmission Weight`, `Coat Weight`, `Emission Color`).
    """
    mat = bpy.data.materials.get(name) if overwrite else None
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (540, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (220, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    rgba_a = _to_rgba(base_color)
    if secondary_color is not None:
        rgba_b = _to_rgba(secondary_color)
    else:
        rgba_b = (
            max(0.0, rgba_a[0] * 0.82),
            max(0.0, rgba_a[1] * 0.82),
            max(0.0, rgba_a[2] * 0.82),
            rgba_a[3],
        )

    pat = str(pattern).strip().upper()
    fac_out = None
    if pat != "FLAT":
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-720, 0)
        coord_key = {
            "OBJECT": "Object",
            "GENERATED": "Generated",
            "UV": "UV",
        }.get(str(coord_mode).strip().upper(), "Object")
        coord_out = tex_coord.outputs[coord_key]

        noise = None
        if pat in ("NOISE", "BOTH"):
            noise = nodes.new("ShaderNodeTexNoise")
            noise.location = (-460, 140)
            noise.inputs["Scale"].default_value = float(pattern_scale)
            noise.inputs["Detail"].default_value = float(pattern_detail)
            noise.inputs["Roughness"].default_value = 0.55
            links.new(coord_out, noise.inputs["Vector"])
            fac_out = noise.outputs["Fac"]

        if pat in ("VORONOI", "BOTH"):
            vor = nodes.new("ShaderNodeTexVoronoi")
            vor.location = (-460, -140)
            vor.inputs["Scale"].default_value = float(pattern_scale)
            links.new(coord_out, vor.inputs["Vector"])
            if pat == "VORONOI":
                fac_out = vor.outputs["Distance"]
            else:
                mix_f = nodes.new("ShaderNodeMix")
                mix_f.data_type = "FLOAT"
                mix_f.location = (-230, 40)
                mix_f.inputs[0].default_value = 0.45
                links.new(noise.outputs["Fac"], mix_f.inputs[2])
                links.new(vor.outputs["Distance"], mix_f.inputs[3])
                fac_out = mix_f.outputs[0]

    if fac_out is not None:
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (-40, 220)
        ramp.color_ramp.elements[0].position = 0.25
        ramp.color_ramp.elements[0].color = rgba_b
        ramp.color_ramp.elements[1].position = 0.78
        ramp.color_ramp.elements[1].color = rgba_a
        links.new(fac_out, ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        if float(roughness_variation) > 0.0:
            map_r = nodes.new("ShaderNodeMapRange")
            map_r.location = (-40, -10)
            map_r.clamp = True
            map_r.inputs["From Min"].default_value = 0.0
            map_r.inputs["From Max"].default_value = 1.0
            map_r.inputs["To Min"].default_value = max(0.02, float(roughness) - float(roughness_variation))
            map_r.inputs["To Max"].default_value = min(1.0, float(roughness) + float(roughness_variation))
            links.new(fac_out, map_r.inputs["Value"])
            links.new(map_r.outputs["Result"], bsdf.inputs["Roughness"])
        else:
            _safe_set_bsdf_input(bsdf, "Roughness", float(roughness))

        if float(bump_strength) > 0.0:
            bump = nodes.new("ShaderNodeBump")
            bump.location = (-40, -240)
            bump.inputs["Strength"].default_value = float(bump_strength)
            bump.inputs["Distance"].default_value = 0.005
            links.new(fac_out, bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        _safe_set_bsdf_input(bsdf, "Base Color", rgba_a)
        _safe_set_bsdf_input(bsdf, "Roughness", float(roughness))

    _safe_set_bsdf_input(bsdf, "Metallic", float(metallic))
    _safe_set_bsdf_input(bsdf, "IOR", float(ior))
    _safe_set_bsdf_input(bsdf, "Specular IOR Level", float(specular_ior_level))
    _safe_set_bsdf_input(bsdf, "Subsurface Weight", float(subsurface_weight))
    _safe_set_bsdf_input(bsdf, "Subsurface Scale", float(subsurface_scale))
    _safe_set_bsdf_input(bsdf, "Subsurface Radius", tuple(float(x) for x in subsurface_radius))
    _safe_set_bsdf_input(bsdf, "Transmission Weight", float(transmission_weight))
    _safe_set_bsdf_input(bsdf, "Coat Weight", float(coat_weight))
    _safe_set_bsdf_input(bsdf, "Coat Roughness", float(coat_roughness))
    _safe_set_bsdf_input(bsdf, "Emission Color", _to_rgba(emission_color))
    _safe_set_bsdf_input(bsdf, "Emission Strength", float(emission_strength))
    _safe_set_bsdf_input(bsdf, "Alpha", float(alpha))

    if (float(alpha) < 0.999 or float(transmission_weight) > 0.0) and hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"

    return mat


def make_xray_material(
    name="XRay_Cutaway",
    rim_color=(0.25, 0.85, 1.0, 1.0),
    core_color=(0.05, 0.25, 0.45, 1.0),
    rim_power=2.8,
    core_alpha=0.10,
    rim_alpha=0.82,
    emission_strength=1.6,
    use_dithered=True,
    overwrite=True,
):
    """
    Build a Fresnel rim-glow X-ray cutaway shader using Blender 4.5
    `surface_render_method = 'DITHERED'` and explicit ShaderNodeMix RGBA/FLOAT
    socket indices (`inputs[6]`/`inputs[7]`/`outputs[2]` for RGBA;
    `inputs[2]`/`inputs[3]`/`outputs[0]` for FLOAT).
    """
    mat = bpy.data.materials.get(name) if overwrite else None
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (620, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    lw = nodes.new("ShaderNodeLayerWeight")
    lw.location = (-380, 0)
    lw.inputs["Blend"].default_value = 0.5

    pwr = nodes.new("ShaderNodeMath")
    pwr.operation = "POWER"
    pwr.use_clamp = True
    pwr.location = (-180, 0)
    pwr.inputs[1].default_value = float(rim_power)
    links.new(lw.outputs["Facing"], pwr.inputs[0])

    mix_col = nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.location = (40, 120)
    mix_col.inputs[6].default_value = _to_rgba(core_color)
    mix_col.inputs[7].default_value = _to_rgba(rim_color)
    links.new(pwr.outputs["Value"], mix_col.inputs[0])

    mix_alpha = nodes.new("ShaderNodeMix")
    mix_alpha.data_type = "FLOAT"
    mix_alpha.location = (40, -120)
    mix_alpha.inputs[2].default_value = float(core_alpha)
    mix_alpha.inputs[3].default_value = float(rim_alpha)
    links.new(pwr.outputs["Value"], mix_alpha.inputs[0])

    _safe_set_bsdf_input(bsdf, "Base Color", _to_rgba(core_color))
    _safe_set_bsdf_input(bsdf, "Roughness", 0.15)
    _safe_set_bsdf_input(bsdf, "Specular IOR Level", 0.2)
    _safe_set_bsdf_input(bsdf, "Emission Strength", float(emission_strength))
    links.new(mix_col.outputs[2], bsdf.inputs["Emission Color"])
    links.new(mix_alpha.outputs[0], bsdf.inputs["Alpha"])

    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED" if use_dithered else "BLENDED"

    return mat


# ==============================================================================
# MODULE 4 — STAGING, CAMERA, & ANIMATION HELPERS (2.3:1 & 2.4:1)
# ==============================================================================

def build_stage(
    anchor_id="prep_counter",
    surface_type="warm_oak_butcher_block",
    lighting_mood="bright_clean_studio",
    scale_tier="tabletop_miniature",
):
    """
    Generate a procedural z = 0 stage floor/countertop, backdrop, and 3-point
    studio lighting rig sized to `scale_tier` ('tabletop_miniature' or
    'industrial_yard').
    """
    scene = bpy.context.scene
    coll = _ensure_collection(f"Stage_{anchor_id}")

    stage_root = bpy.data.objects.new(f"Stage_{anchor_id}_Root", None)
    stage_root.empty_display_type = "PLAIN_AXES"
    coll.objects.link(stage_root)

    is_industrial = str(scale_tier).strip().lower() == "industrial_yard"
    span = 24.0 if is_industrial else 2.6
    thick = 0.40 if is_industrial else 0.06
    wall_h = 14.0 if is_industrial else 1.8
    light_dist = 10.0 if is_industrial else 1.6
    light_scale = 28.0 if is_industrial else 1.0

    mood = str(lighting_mood).strip().lower()
    surf = str(surface_type).strip().lower()
    is_xray_void = ("xray" in str(anchor_id).lower()) or ("xray" in surf) or ("void" in surf)

    if scene.world is None:
        scene.world = bpy.data.worlds.new(f"World_{anchor_id}")
    scene.world.use_nodes = True
    bg = next((n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        if is_xray_void or "dark" in mood:
            bg.inputs[0].default_value = (0.018, 0.024, 0.036, 1.0)
            bg.inputs[1].default_value = 0.35
        else:
            bg.inputs[0].default_value = (0.14, 0.17, 0.21, 1.0)
            bg.inputs[1].default_value = 0.85

    floor_obj = None
    backdrop_obj = None

    if not is_xray_void:
        bm_f = bmesh.new()
        bmesh.ops.create_cube(bm_f, size=1.0)
        for v in bm_f.verts:
            v.co.x *= span
            v.co.y *= span
            v.co.z = (v.co.z - 0.5) * thick
        f_mesh = bpy.data.meshes.new(f"Stage_{anchor_id}_FloorMesh")
        bm_f.to_mesh(f_mesh)
        bm_f.free()

        if "steel" in surf or "metal" in surf:
            f_mat = make_pbr_material(
                name=f"Mat_{anchor_id}_Floor",
                base_color=(0.62, 0.64, 0.67, 1.0),
                secondary_color=(0.50, 0.52, 0.55, 1.0),
                metallic=0.88,
                roughness=0.28,
                roughness_variation=0.06,
                pattern="NOISE",
                pattern_scale=16.0,
                bump_strength=0.02,
            )
        elif "stone" in surf or "quarry" in surf or "concrete" in surf or is_industrial:
            f_mat = make_pbr_material(
                name=f"Mat_{anchor_id}_Floor",
                base_color=(0.24, 0.24, 0.25, 1.0),
                secondary_color=(0.15, 0.15, 0.16, 1.0),
                roughness=0.82,
                roughness_variation=0.10,
                pattern="BOTH",
                pattern_scale=6.0 if is_industrial else 14.0,
                bump_strength=0.14,
            )
        else:
            f_mat = make_pbr_material(
                name=f"Mat_{anchor_id}_Floor",
                base_color=(0.48, 0.30, 0.16, 1.0),
                secondary_color=(0.34, 0.19, 0.09, 1.0),
                roughness=0.48,
                roughness_variation=0.10,
                pattern="NOISE",
                pattern_scale=10.0,
                bump_strength=0.04,
            )

        f_mesh.materials.append(f_mat)
        floor_obj = bpy.data.objects.new(f"Stage_{anchor_id}_Floor", f_mesh)
        coll.objects.link(floor_obj)
        floor_obj.parent = stage_root

        bm_b = bmesh.new()
        bmesh.ops.create_cube(bm_b, size=1.0)
        for v in bm_b.verts:
            v.co.x *= span * 1.4
            v.co.y = v.co.y * 0.05 + (span * 0.48)
            v.co.z = (v.co.z + 0.5) * wall_h - thick
        b_mesh = bpy.data.meshes.new(f"Stage_{anchor_id}_WallMesh")
        bm_b.to_mesh(b_mesh)
        bm_b.free()

        w_mat = make_pbr_material(
            name=f"Mat_{anchor_id}_Wall",
            base_color=(0.24, 0.36, 0.48, 1.0) if "bright" in mood else (0.12, 0.14, 0.18, 1.0),
            roughness=0.85,
            roughness_variation=0.0,
            pattern="FLAT",
        )
        b_mesh.materials.append(w_mat)
        backdrop_obj = bpy.data.objects.new(f"Stage_{anchor_id}_Backdrop", b_mesh)
        coll.objects.link(backdrop_obj)
        backdrop_obj.parent = stage_root

    aim_target = bpy.data.objects.new(f"Stage_{anchor_id}_LightAim", None)
    aim_target.location = (0.0, 0.0, 0.6 if is_industrial else 0.12)
    coll.objects.link(aim_target)
    aim_target.parent = stage_root

    lights = []
    light_specs = [
        ("Key_Light", (light_dist * 0.85, -light_dist * 0.90, light_dist * 1.10), 65.0 * light_scale, (1.0, 0.97, 0.92), light_dist * 0.6),
        ("Fill_Light", (-light_dist * 0.95, -light_dist * 0.70, light_dist * 0.65), 28.0 * light_scale, (0.86, 0.92, 1.0), light_dist * 0.9),
        ("Rim_Light", (0.0, light_dist * 0.85, light_dist * 0.95), 42.0 * light_scale, (0.95, 0.98, 1.0), light_dist * 0.5),
    ]
    for l_name, l_pos, l_energy, l_col, l_size in light_specs:
        ldata = bpy.data.lights.new(name=f"{anchor_id}_{l_name}", type="AREA")
        ldata.energy = float(l_energy)
        ldata.color = l_col
        ldata.size = float(l_size)
        lobj = bpy.data.objects.new(name=f"{anchor_id}_{l_name}", object_data=ldata)
        lobj.location = l_pos
        coll.objects.link(lobj)
        lobj.parent = stage_root
        c = lobj.constraints.new(type="TRACK_TO")
        c.target = aim_target
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"
        lights.append(lobj)

    bpy.context.view_layer.update()
    return {
        "stage_root": stage_root,
        "floor": floor_obj,
        "backdrop": backdrop_obj,
        "lights": lights,
    }


def setup_track_to_camera(
    focal_point=(0.0, 0.0, 0.10),
    camera_pos=(0.0, -0.75, 0.36),
    lens_mm=50.0,
    safe_zone_offset_z=0.025,
    target_name="Focal_Target_Empty",
    camera_name="Shorts_Camera",
):
    """
    Create or configure the 1080x1920 camera constrained via `TRACK_TO`
    (`TRACK_NEGATIVE_Z`, `UP_Y`) onto `Focal_Target_Empty`.

    `safe_zone_offset_z` aims slightly below `focal_point.z` so the primary
    subject rides in the upper-middle YouTube Shorts safe zone, clear of the
    bottom 20% captions/title bar.
    """
    scene = bpy.context.scene
    coll = scene.collection

    target_empty = bpy.data.objects.get(target_name)
    if target_empty is None:
        target_empty = bpy.data.objects.new(target_name, None)
        target_empty.empty_display_type = "SPHERE"
        target_empty.empty_display_size = 0.06
        coll.objects.link(target_empty)

    fp = mathutils.Vector(focal_point)
    target_empty.location = (fp.x, fp.y, fp.z - float(safe_zone_offset_z))

    cam_obj = bpy.data.objects.get(camera_name)
    if cam_obj is None or cam_obj.type != "CAMERA":
        cam_data = bpy.data.cameras.new(camera_name)
        cam_obj = bpy.data.objects.new(camera_name, cam_data)
        coll.objects.link(cam_obj)
    else:
        cam_data = cam_obj.data

    cam_data.type = "PERSP"
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = 36.0
    cam_data.lens = float(lens_mm)
    cam_data.clip_start = 0.005
    cam_data.clip_end = 500.0

    cam_obj.location = tuple(float(v) for v in camera_pos)

    for c in list(cam_obj.constraints):
        if c.type == "TRACK_TO":
            cam_obj.constraints.remove(c)

    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = target_empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene.camera = cam_obj
    bpy.context.view_layer.update()
    return cam_obj, target_empty


def _apply_fcurve_easing(id_block, data_path, frame, interpolation="EXPO", easing="EASE_OUT"):
    """Apply classic Blender 4.5 Action.fcurves interpolation & easing at `frame`."""
    ad = getattr(id_block, "animation_data", None)
    if ad is None or ad.action is None:
        return
    interp = str(interpolation).strip().upper()
    ease = str(easing).strip().upper()
    target_f = float(frame)
    for fc in ad.action.fcurves:
        if fc.data_path != data_path:
            continue
        for kp in fc.keyframe_points:
            if abs(kp.co.x - target_f) < 0.01:
                kp.interpolation = interp
                if interp not in ("CONSTANT", "LINEAR", "BEZIER"):
                    kp.easing = ease


def keyframe_prop(
    obj,
    frame,
    location=None,
    rotation_euler=None,
    scale=None,
    hide_render=None,
    hide_viewport=None,
    interpolation="EXPO",
    easing="EASE_OUT",
):
    """Keyframe rigid transforms or visibility swaps using classic Action.fcurves."""
    f = int(frame)
    if location is not None:
        obj.location = tuple(float(v) for v in location)
        obj.keyframe_insert(data_path="location", frame=f)
        _apply_fcurve_easing(obj, "location", f, interpolation, easing)

    if rotation_euler is not None:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = tuple(float(v) for v in rotation_euler)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)
        _apply_fcurve_easing(obj, "rotation_euler", f, interpolation, easing)

    if scale is not None:
        obj.scale = tuple(float(v) for v in scale)
        obj.keyframe_insert(data_path="scale", frame=f)
        _apply_fcurve_easing(obj, "scale", f, interpolation, easing)

    if hide_render is not None:
        obj.hide_render = bool(hide_render)
        obj.keyframe_insert(data_path="hide_render", frame=f)
        _apply_fcurve_easing(obj, "hide_render", f, "CONSTANT", "EASE_OUT")

    if hide_viewport is not None:
        obj.hide_viewport = bool(hide_viewport)
        obj.keyframe_insert(data_path="hide_viewport", frame=f)
        _apply_fcurve_easing(obj, "hide_viewport", f, "CONSTANT", "EASE_OUT")

    return obj


def keyframe_shape_key(obj, key_name, frame, value, interpolation="CUBIC", easing="EASE_OUT"):
    """Keyframe an authored Shape Key `key_name` (`0.0` to `1.0`) at `frame`."""
    mesh_obj = _resolve_mesh_object(obj)
    sk = mesh_obj.data.shape_keys
    if sk is None or key_name not in sk.key_blocks:
        raise ValueError(f"Shape Key '{key_name}' not found on '{mesh_obj.name}'.")

    f = int(frame)
    kb = sk.key_blocks[key_name]
    kb.value = float(value)
    kb.keyframe_insert(data_path="value", frame=f)
    _apply_fcurve_easing(sk, f'key_blocks["{key_name}"].value', f, interpolation, easing)
    return kb


def bake_all_physics(start_frame=1, end_frame=None):
    """
    Bake all scene physics caches (Rigid Body / Particle / Smoke) strictly from
    frame 1 so out-of-order still checks or clipped renders never freeze.
    """
    scene = bpy.context.scene
    scene.frame_start = int(start_frame)
    if end_frame is not None:
        scene.frame_end = int(end_frame)

    has_rb = scene.rigidbody_world is not None
    has_ps = any(len(obj.particle_systems) > 0 for obj in bpy.data.objects)
    has_sim_mod = any(
        m.type in {"CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT"}
        for obj in bpy.data.objects
        for m in obj.modifiers
    )

    if not (has_rb or has_ps or has_sim_mod):
        scene.frame_set(int(start_frame))
        return False

    if has_rb:
        scene.rigidbody_world.point_cache.frame_start = int(start_frame)
        scene.rigidbody_world.point_cache.frame_end = int(scene.frame_end)

    bpy.context.view_layer.update()
    try:
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all(bake=True)
    except RuntimeError:
        for obj in bpy.data.objects:
            for ps in obj.particle_systems:
                ps.point_cache.frame_start = int(start_frame)
                ps.point_cache.frame_end = int(scene.frame_end)
        bpy.ops.ptcache.bake_all(bake=True)

    scene.frame_set(int(start_frame))
    return True


# ==============================================================================
# MODULE 5 — RAPID PREVIEW RENDERING (2.2:3, 2.3:2, 2.4:2)
# ==============================================================================

def render_stills(
    items,
    out_dir,
    width=540,
    height=960,
    samples=8,
    filename_prefix="still",
    target_obj=None,
):
    """
    Render fast low-sample preview PNGs for vision contact-sheet verification.
    Automatically injects a temporary preview Sun + ambient World when run on
    an unlit asset .blend (such as Block 2.2:3 turnaround checks) and cleans
    them up in `finally`.

    `items` accepts either:
      - Timeline frames (`[15, 56, 120]` or `[{"frame": 56, "name": "cue_01_02"}]`)
        for Block 2.3:2 static layout and Block 2.4:2 cue-midpoint checks.
      - `"turnaround"` (auto-places 4 angles around `target_obj` for Block 2.2:3)
        or explicit camera specs (`[{"name": "front", "cam_pos": ..., "target": ...}]`).

    Returns:
        list[str]: Rendered PNG file paths in order.
    """
    scene = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)

    prev_res_x = scene.render.resolution_x
    prev_res_y = scene.render.resolution_y
    prev_pct = scene.render.resolution_percentage
    prev_fmt = scene.render.image_settings.file_format
    prev_samples = getattr(scene.eevee, "taa_render_samples", 16)
    prev_filepath = scene.render.filepath
    prev_frame = scene.frame_current
    prev_cam = scene.camera
    prev_world = scene.world

    rendered_paths = []
    temp_cam_obj = None
    temp_sun_obj = None
    temp_fill_obj = None
    temp_world = None

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.resolution_x = int(width)
        scene.render.resolution_y = int(height)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = int(samples)

        # Ensure turnaround/asset checks on unlit .blend files never render pitch black
        if not any(o.type == "LIGHT" and not o.hide_render for o in scene.objects):
            sun_data = bpy.data.lights.new(name="_PreviewKeySun", type="SUN")
            sun_data.energy = 2.8
            sun_data.color = (1.0, 0.98, 0.95)
            sun_data.angle = math.radians(15.0)
            temp_sun_obj = bpy.data.objects.new(name="_PreviewKeySun", object_data=sun_data)
            scene.collection.objects.link(temp_sun_obj)
            temp_sun_obj.rotation_euler = mathutils.Vector((1.2, -1.4, 1.8)).normalized().to_track_quat("Z", "Y").to_euler()

            fill_data = bpy.data.lights.new(name="_PreviewFillSun", type="SUN")
            fill_data.energy = 1.4
            fill_data.color = (0.88, 0.93, 1.0)
            fill_data.angle = math.radians(25.0)
            temp_fill_obj = bpy.data.objects.new(name="_PreviewFillSun", object_data=fill_data)
            scene.collection.objects.link(temp_fill_obj)
            temp_fill_obj.rotation_euler = mathutils.Vector((-1.1, 1.4, 1.0)).normalized().to_track_quat("Z", "Y").to_euler()

        if scene.world is None:
            temp_world = bpy.data.worlds.new("_PreviewWorld")
            temp_world.use_nodes = True
            bg = next((n for n in temp_world.node_tree.nodes if n.type == "BACKGROUND"), None)
            if bg is not None:
                bg.inputs[0].default_value = (0.06, 0.07, 0.09, 1.0)
                bg.inputs[1].default_value = 0.85
            scene.world = temp_world

        if items == "turnaround":
            if target_obj is None:
                raise ValueError("items='turnaround' requires `target_obj`.")
            b_min, b_max = get_world_bounds(target_obj)
            center = 0.5 * (b_min + b_max)
            dims = b_max - b_min
            dist = max(dims.length * 1.45, 0.25)
            items = [
                {"name": "front", "cam_pos": (center.x, center.y - dist, center.z + dims.z * 0.10), "target": tuple(center)},
                {"name": "back", "cam_pos": (center.x, center.y + dist, center.z + dims.z * 0.10), "target": tuple(center)},
                {"name": "side", "cam_pos": (center.x + dist, center.y, center.z + dims.z * 0.10), "target": tuple(center)},
                {"name": "iso", "cam_pos": (center.x + dist * 0.72, center.y - dist * 0.85, center.z + dist * 0.55), "target": tuple(center)},
            ]

        for idx, entry in enumerate(items):
            if isinstance(entry, int):
                frame_no = entry
                out_name = f"{filename_prefix}_{idx:02d}_f{frame_no:04d}.png"
                scene.frame_set(frame_no)
                bpy.context.view_layer.update()
            elif isinstance(entry, dict) and "frame" in entry:
                frame_no = int(entry["frame"])
                tag = entry.get("name") or f"f{frame_no:04d}"
                out_name = f"{filename_prefix}_{idx:02d}_{tag}.png"
                scene.frame_set(frame_no)
                bpy.context.view_layer.update()
            elif isinstance(entry, dict) and "cam_pos" in entry:
                if temp_cam_obj is None:
                    cam_data = bpy.data.cameras.new("_PreviewStillCam")
                    cam_data.lens = 50.0
                    cam_data.clip_start = 0.005
                    temp_cam_obj = bpy.data.objects.new("_PreviewStillCam", cam_data)
                    scene.collection.objects.link(temp_cam_obj)
                scene.camera = temp_cam_obj
                cam_pos = mathutils.Vector(entry["cam_pos"])
                tgt = mathutils.Vector(entry["target"])
                temp_cam_obj.location = cam_pos
                direction = (tgt - cam_pos).normalized()
                temp_cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
                tag = entry.get("name") or f"view_{idx:02d}"
                out_name = f"{filename_prefix}_{idx:02d}_{tag}.png"
                bpy.context.view_layer.update()
            else:
                raise ValueError(f"Unsupported render_stills item: {entry!r}")

            out_path = os.path.abspath(os.path.join(out_dir, out_name))
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            rendered_paths.append(out_path)
    finally:
        scene.camera = prev_cam
        scene.world = prev_world
        if temp_world is not None and temp_world.users == 0:
            bpy.data.worlds.remove(temp_world)
        for t_light in (temp_sun_obj, temp_fill_obj):
            if t_light is not None:
                l_data = t_light.data
                bpy.data.objects.remove(t_light, do_unlink=True)
                if l_data and l_data.users == 0:
                    bpy.data.lights.remove(l_data)
        if temp_cam_obj is not None:
            cam_data = temp_cam_obj.data
            bpy.data.objects.remove(temp_cam_obj, do_unlink=True)
            if cam_data and cam_data.users == 0:
                bpy.data.cameras.remove(cam_data)
        scene.render.resolution_x = prev_res_x
        scene.render.resolution_y = prev_res_y
        scene.render.resolution_percentage = prev_pct
        scene.render.image_settings.file_format = prev_fmt
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = prev_samples
        scene.render.filepath = prev_filepath
        scene.frame_set(prev_frame)

    return rendered_paths