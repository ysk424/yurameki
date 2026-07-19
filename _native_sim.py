"""Blender orchestration for the Yurameki C++/OpenMP simulation core.

All numerical strand work lives in ``_yurameki_native_0_3_0``.  This module only
touches Blender RNA/depsgraph data, controls frames, and stores output caches.
Blender data is deliberately never accessed from an OpenMP worker thread.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import bpy
import numpy as np
from bpy.app.handlers import persistent

if __package__:
    from . import _yurameki_native_0_3_0 as native
else:
    import _yurameki_native_0_3_0 as native


STRETCH_STIFFNESS = 1.0e4


@dataclass
class NativeCheckStats:
    n_strands: int
    points_per_strand: int
    n_points: int
    n_segments: int
    root_locked_points: int
    n_vertices: int
    n_triangles: int
    device: str
    device_name: str
    native_version: str


@dataclass
class SimStats:
    start_frame: int
    end_frame: int
    n_frames: int
    n_strands: int
    simulated_strands: int
    guide_decimation: int
    points_per_strand: int
    root_locked_points: int
    adaptive_root_lock: bool
    adaptive_lock_max_points: int
    adaptive_lock_strand_frames: int
    frame_steps: int
    total_substeps: int
    max_substeps: int
    max_auto_move_mm: float
    total_hits: int
    n_triangles_last: int
    bake_mode: str
    max_velocity_mps: float
    collision_max_correction_mm: float
    collision_response: float
    collision_velocity_damping: float
    keep_length: bool
    keep_length_source_frame: int
    max_keep_length_error_mm: float
    cache_path: str
    device: str
    device_name: str
    elapsed_sec: float
    groomed_strand_frames: int
    settle_failed_strand_frames: int
    length_bad_strand_frames: int
    length_bad_rod_frames: int
    folded_strand_frames: int
    max_settle_iterations: int
    surface_feedback_repairs: int
    shape_bad_strand_frames: int
    rough_strand_frames: int
    collision_bad_strand_frames: int
    unresolved_shape_strand_frames: int
    unresolved_collision_strand_frames: int
    max_pre_groom_length_error_mm: float


@dataclass
class YuramekiRuntimeCache:
    object_name: str
    data_name: str
    blend_path: str
    frames: np.ndarray
    local_values: np.ndarray
    restore_local_values: np.ndarray
    curve_spans: np.ndarray
    path: str

    @property
    def start_frame(self) -> int:
        return int(self.frames[0])

    @property
    def end_frame(self) -> int:
        return int(self.frames[-1])

    @property
    def n_points(self) -> int:
        return int(self.local_values.shape[1])


_CACHE_REGISTRY: dict[str, YuramekiRuntimeCache] = {}
_CACHE_MUTED = False


def _curve_spans(curves_obj) -> list[tuple[int, int]]:
    return [
        (int(curve.first_point_index), int(curve.points_length))
        for curve in curves_obj.data.curves
    ]


def _uniform_points_per_strand(curves_obj) -> tuple[int, int]:
    spans = _curve_spans(curves_obj)
    if not spans:
        raise ValueError("Curves object has no strands")
    sizes = sorted({size for _start, size in spans})
    if len(sizes) != 1:
        raise ValueError(f"C++ path requires uniform points per strand, got {sizes[:8]}")
    if sizes[0] < 3:
        raise ValueError("C++ path needs at least 3 points per strand")
    return sizes[0], len(spans)


def _read_local_points(curves_obj) -> np.ndarray:
    attr = curves_obj.data.attributes.get("position")
    if attr is None or len(attr.data) == 0:
        raise ValueError("Curves has no position attribute")
    flat = np.empty(len(attr.data) * 3, dtype=np.float32)
    attr.data.foreach_get("vector", flat)
    return np.ascontiguousarray(flat.reshape(-1, 3), dtype=np.float32)


def _write_local_points(curves_obj, local_points: np.ndarray) -> None:
    local = np.ascontiguousarray(local_points, dtype=np.float32)
    attr = curves_obj.data.attributes.get("position")
    if attr is None or local.shape != (len(attr.data), 3):
        raise ValueError("Curves position attribute shape changed")
    attr.data.foreach_set("vector", local.ravel())
    curves_obj.data.update_tag()


def _read_world_points(data_owner, expected: int, matrix_world):
    attr = data_owner.attributes.get("position")
    if attr is None or len(attr.data) != expected:
        return None
    flat = np.empty(expected * 3, dtype=np.float32)
    attr.data.foreach_get("vector", flat)
    local = flat.reshape(-1, 3)
    matrix = np.asarray(matrix_world, dtype=np.float32)
    homogeneous = np.column_stack((local, np.ones(expected, dtype=np.float32)))
    return np.ascontiguousarray((homogeneous @ matrix.T)[:, :3], dtype=np.float32)


def _read_world(curves_obj) -> tuple[np.ndarray, np.ndarray]:
    attr = curves_obj.data.attributes.get("position")
    if attr is None or len(attr.data) == 0:
        raise ValueError("Curves has no points")
    expected = len(attr.data)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = curves_obj.evaluated_get(depsgraph)
    evaluated_world = _read_world_points(evaluated.data, expected, evaluated.matrix_world)
    original_world = _read_world_points(curves_obj.data, expected, curves_obj.matrix_world)
    if evaluated_world is None or original_world is None:
        raise ValueError("Could not read evaluated Curves positions")
    return evaluated_world, original_world


def _world_to_local_points(curves_obj, world_points: np.ndarray, offset=None) -> np.ndarray:
    world = np.asarray(world_points, dtype=np.float32)
    write_world = world if offset is None else world - np.asarray(offset, dtype=np.float32)
    inverse = np.asarray(curves_obj.matrix_world.inverted(), dtype=np.float32)
    homogeneous = np.column_stack((write_world, np.ones(len(world), dtype=np.float32)))
    return np.ascontiguousarray((homogeneous @ inverse.T)[:, :3], dtype=np.float32)


def _write_world_points(curves_obj, world_points: np.ndarray, offset=None) -> None:
    _write_local_points(curves_obj, _world_to_local_points(curves_obj, world_points, offset))


def _force_viewport_refresh() -> None:
    bpy.context.view_layer.update()
    if bpy.app.background:
        return
    for window in bpy.context.window_manager.windows:
        if window.screen is None:
            continue
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _evaluated_mesh_arrays(objects) -> tuple[np.ndarray, np.ndarray]:
    vertices_out: list[np.ndarray] = []
    triangles_out: list[np.ndarray] = []
    vertex_offset = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            if len(mesh.vertices) == 0 or len(mesh.loop_triangles) == 0:
                continue
            local = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", local)
            local = local.reshape(-1, 3)
            matrix = np.asarray(evaluated.matrix_world, dtype=np.float32)
            homogeneous = np.column_stack((local, np.ones(len(local), dtype=np.float32)))
            world = np.ascontiguousarray((homogeneous @ matrix.T)[:, :3], dtype=np.float32)
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", triangles)
            triangles = triangles.reshape(-1, 3)
            # A reflected object matrix reverses the cross-product winding in
            # world space.  Swap two vertices so Body signed normals remain
            # outward under a negative object scale.
            if float(np.linalg.det(matrix[:3, :3])) < 0.0:
                triangles = triangles[:, (0, 2, 1)]
            triangles = triangles + vertex_offset
            vertices_out.append(world)
            triangles_out.append(triangles)
            vertex_offset += len(world)
        finally:
            evaluated.to_mesh_clear()
    if not vertices_out:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    return (
        np.ascontiguousarray(np.vstack(vertices_out), dtype=np.float32),
        np.ascontiguousarray(np.vstack(triangles_out), dtype=np.int32),
    )


def _armature_from_object(obj):
    if obj is None:
        return None
    if getattr(obj, "parent", None) is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    for modifier in getattr(obj, "modifiers", ()):
        if modifier.type == "ARMATURE" and getattr(modifier, "object", None) is not None:
            return modifier.object
    return None


def _head_bone_name(armature):
    for name in ("CC_Base_Head", "Head", "head"):
        if name in armature.pose.bones or name in armature.data.bones:
            return name
    for bone in armature.data.bones:
        if "head" in bone.name.lower():
            return bone.name
    return None


def _ear_anchor_world(armature):
    for name in ("CC_Base_JawRoot", "JawRoot"):
        bone = armature.pose.bones.get(name)
        if bone is not None:
            return np.asarray(armature.matrix_world @ bone.head, dtype=np.float32)
    eyes = [armature.pose.bones.get(name) for name in ("CC_Base_L_Eye", "CC_Base_R_Eye")]
    eyes = [bone for bone in eyes if bone is not None]
    if not eyes:
        return None
    return np.mean(
        [np.asarray(armature.matrix_world @ bone.head, dtype=np.float32) for bone in eyes],
        axis=0,
    ).astype(np.float32)


def _head_frame_world(body_obj):
    armature = _armature_from_object(body_obj)
    if armature is None:
        return None
    name = _head_bone_name(armature)
    if name is None:
        return None
    pose_bone = armature.pose.bones.get(name)
    if pose_bone is not None:
        base = np.asarray(armature.matrix_world @ pose_bone.head, dtype=np.float32)
        tip = np.asarray(armature.matrix_world @ pose_bone.tail, dtype=np.float32)
    else:
        bone = armature.data.bones.get(name)
        if bone is None:
            return None
        base = np.asarray(armature.matrix_world @ bone.head_local, dtype=np.float32)
        tip = np.asarray(armature.matrix_world @ bone.tail_local, dtype=np.float32)
    up = tip - base
    norm = float(np.linalg.norm(up))
    up = up / norm if norm > 1.0e-9 else np.array((0.0, 0.0, 1.0), dtype=np.float32)
    ear = _ear_anchor_world(armature)
    ear_offset = float(np.dot(ear - base, up)) if ear is not None else 0.0
    return {"base": base, "up": up, "tip": tip, "ear_offset": ear_offset}


def _read_targets_for_frames(curves_obj, frames: list[int], restore_local: np.ndarray):
    worlds: dict[int, np.ndarray] = {}
    offsets: dict[int, np.ndarray] = {}
    locals_by_frame: dict[int, np.ndarray] = {}
    expected = None
    scene = bpy.context.scene
    for frame in frames:
        # Remove the preceding preview result before evaluating the next pose.
        # frame_set may then legitimately replace it with existing keyframes.
        _write_local_points(curves_obj, restore_local)
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        evaluated, original = _read_world(curves_obj)
        if expected is None:
            expected = len(evaluated)
        elif len(evaluated) != expected:
            raise ValueError(f"Curves topology changed at frame {frame}")
        worlds[frame] = evaluated.copy()
        offsets[frame] = evaluated - original
        locals_by_frame[frame] = _read_local_points(curves_obj)
    return worlds, offsets, locals_by_frame


def _bake_local_position_keyframes(curves_obj, frames: list[int], local_values: np.ndarray) -> None:
    attr = curves_obj.data.attributes.get("position")
    if attr is None or local_values.shape != (len(frames), len(attr.data), 3):
        raise ValueError("Cached position shape does not match Curves topology")
    data = curves_obj.data
    animation = data.animation_data_create()
    if animation.action is None:
        animation.action = bpy.data.actions.new(f"Yurameki Bake {curves_obj.name}")
    action = animation.action
    frame_numbers = np.asarray(frames, dtype=np.float32)
    coordinates = np.empty(len(frames) * 2, dtype=np.float32)
    coordinates[0::2] = frame_numbers
    interpolation = np.ones(len(frames), dtype=np.int32)
    for point_index in range(len(attr.data)):
        data_path = f'attributes["position"].data[{point_index}].vector'
        for axis in range(3):
            fcurve = action.fcurve_ensure_for_datablock(
                data, data_path, index=axis, group_name="Yurameki Position"
            )
            fcurve.keyframe_points.clear()
            fcurve.keyframe_points.add(len(frames))
            coordinates[1::2] = local_values[:, point_index, axis]
            fcurve.keyframe_points.foreach_set("co", coordinates)
            fcurve.keyframe_points.foreach_set("interpolation", interpolation)
            fcurve.update()
    bpy.context.scene.frame_set(frames[-1])
    bpy.context.view_layer.update()


def _cache_dir() -> str:
    path = bpy.path.abspath("//yurameki_cache") if bpy.data.filepath else os.path.join(tempfile.gettempdir(), "yurameki_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_file_path(curves_obj, start_frame: int, end_frame: int) -> str:
    blend_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] if bpy.data.filepath else f"unsaved_{os.getpid()}"
    safe_blend = "".join(char if char.isalnum() or char in "._-" else "_" for char in blend_name)
    safe_object = "".join(char if char.isalnum() or char in "._-" else "_" for char in curves_obj.name)
    return os.path.join(_cache_dir(), f"{safe_blend}_{safe_object}_{start_frame}_{end_frame}.npz")


def _current_blend_path() -> str:
    return os.path.abspath(bpy.data.filepath) if bpy.data.filepath else ""


def _cache_object(cache: YuramekiRuntimeCache):
    obj = bpy.data.objects.get(cache.object_name)
    if obj is None or obj.type != "CURVES" or obj.data.name != cache.data_name:
        return None
    attr = obj.data.attributes.get("position")
    spans = np.asarray(_curve_spans(obj), dtype=np.int64)
    if (
        attr is None
        or cache.frames.ndim != 1
        or len(cache.frames) == 0
        or cache.local_values.ndim != 3
        or cache.local_values.shape[0] != len(cache.frames)
        or cache.local_values.shape[2] != 3
        or cache.restore_local_values.shape != cache.local_values.shape[1:]
        or len(attr.data) != cache.n_points
        or spans.shape != cache.curve_spans.shape
    ):
        return None
    return obj if np.array_equal(spans, cache.curve_spans) else None


def _remove_cache_metadata(data) -> None:
    for key in ("yurameki_cache_path", "yurameki_cache_start", "yurameki_cache_end"):
        try:
            del data[key]
        except Exception:
            pass


def _apply_cache_frame(cache: YuramekiRuntimeCache, frame: int) -> bool:
    matches = np.flatnonzero(cache.frames == int(frame))
    obj = _cache_object(cache)
    if len(matches) == 0 or obj is None:
        return False
    _write_local_points(obj, cache.local_values[int(matches[0])])
    return True


def _register_sim_cache(curves_obj, frames: list[int], local_values: np.ndarray,
                        restore_local_values: np.ndarray) -> YuramekiRuntimeCache:
    frame_array = np.ascontiguousarray(frames, dtype=np.int32)
    values = np.ascontiguousarray(local_values, dtype=np.float32)
    restore = np.ascontiguousarray(restore_local_values, dtype=np.float32)
    if len(frame_array) == 0 or values.ndim != 3 or values.shape[0] != len(frame_array) or restore.shape != values.shape[1:]:
        raise ValueError("Runtime cache arrays have inconsistent shapes")
    spans = np.ascontiguousarray(_curve_spans(curves_obj), dtype=np.int64)
    blend_path = _current_blend_path()
    path = os.path.abspath(_cache_file_path(curves_obj, int(frame_array[0]), int(frame_array[-1])))
    np.savez(
        path, frames=frame_array, local_values=values,
        restore_local_values=restore, curve_spans=spans,
        object_name=np.array([curves_obj.name]), data_name=np.array([curves_obj.data.name]),
        blend_path=np.array([blend_path]),
    )
    cache = YuramekiRuntimeCache(
        curves_obj.name, curves_obj.data.name, blend_path, frame_array,
        values, restore, spans, path,
    )
    _CACHE_REGISTRY[curves_obj.name] = cache
    curves_obj.data["yurameki_cache_path"] = path
    curves_obj.data["yurameki_cache_start"] = cache.start_frame
    curves_obj.data["yurameki_cache_end"] = cache.end_frame
    register_cache_handler()
    _apply_cache_frame(cache, int(bpy.context.scene.frame_current))
    return cache


def clear_runtime_cache(curves_obj) -> None:
    if curves_obj is None:
        return
    cache = _CACHE_REGISTRY.pop(curves_obj.name, None)
    if cache is not None and _cache_object(cache) is not None:
        _write_local_points(curves_obj, cache.restore_local_values)
    _remove_cache_metadata(curves_obj.data)
    if not _CACHE_REGISTRY:
        unregister_cache_handler()


def get_runtime_cache(curves_obj) -> YuramekiRuntimeCache | None:
    if curves_obj is None:
        return None
    cached = _CACHE_REGISTRY.get(curves_obj.name)
    if cached is not None:
        if _cache_object(cached) is not None and os.path.normcase(cached.blend_path) == os.path.normcase(_current_blend_path()):
            return cached
        _CACHE_REGISTRY.pop(curves_obj.name, None)
    path = str(curves_obj.data.get("yurameki_cache_path", "")).strip()
    if not path or not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            cache = YuramekiRuntimeCache(
                str(data["object_name"][0]), str(data["data_name"][0]), str(data["blend_path"][0]),
                np.ascontiguousarray(data["frames"], dtype=np.int32),
                np.ascontiguousarray(data["local_values"], dtype=np.float32),
                np.ascontiguousarray(data["restore_local_values"], dtype=np.float32),
                np.ascontiguousarray(data["curve_spans"], dtype=np.int64), os.path.abspath(path),
            )
    except Exception:
        return None
    if cache.object_name != curves_obj.name or cache.data_name != curves_obj.data.name:
        return None
    if os.path.normcase(cache.blend_path) != os.path.normcase(_current_blend_path()) or _cache_object(cache) is None:
        return None
    _CACHE_REGISTRY[curves_obj.name] = cache
    register_cache_handler()
    return cache


def bake_runtime_cache(curves_obj) -> dict:
    cache = get_runtime_cache(curves_obj)
    if cache is None:
        raise ValueError("No Yurameki runtime cache for this Curves object")
    frames = [int(frame) for frame in cache.frames]
    _bake_local_position_keyframes(curves_obj, frames, cache.local_values)
    return {
        "object": cache.object_name, "start_frame": cache.start_frame,
        "end_frame": cache.end_frame, "n_frames": len(frames),
        "n_points": cache.n_points, "fcurves": cache.n_points * 3,
        "keys": cache.n_points * 3 * len(frames), "path": cache.path,
    }


@persistent
def _yurameki_cache_frame_change_pre(_scene):
    if _CACHE_MUTED:
        return
    for key, cache in tuple(_CACHE_REGISTRY.items()):
        try:
            obj = _cache_object(cache)
            if obj is None:
                _CACHE_REGISTRY.pop(key, None)
                continue
            _write_local_points(obj, cache.restore_local_values)
        except Exception as exc:
            _CACHE_REGISTRY.pop(key, None)
            print(f"Yurameki cache pre-handler disabled {key}: {exc!r}")


@persistent
def _yurameki_cache_frame_change(scene):
    if _CACHE_MUTED:
        return
    for key, cache in tuple(_CACHE_REGISTRY.items()):
        try:
            if _cache_object(cache) is None:
                _CACHE_REGISTRY.pop(key, None)
                continue
            _apply_cache_frame(cache, int(scene.frame_current))
        except Exception as exc:
            _CACHE_REGISTRY.pop(key, None)
            print(f"Yurameki cache post-handler disabled {key}: {exc!r}")


@persistent
def _yurameki_cache_load_pre(_dummy):
    _CACHE_REGISTRY.clear()


def register_cache_handler() -> None:
    for handlers, function in (
        (bpy.app.handlers.frame_change_pre, _yurameki_cache_frame_change_pre),
        (bpy.app.handlers.frame_change_post, _yurameki_cache_frame_change),
        (bpy.app.handlers.load_pre, _yurameki_cache_load_pre),
    ):
        if function not in handlers:
            handlers.append(function)


def unregister_cache_handler() -> None:
    for cache in tuple(_CACHE_REGISTRY.values()):
        obj = _cache_object(cache)
        if obj is not None:
            try:
                _write_local_points(obj, cache.restore_local_values)
            except Exception:
                pass
    for handlers, function in (
        (bpy.app.handlers.frame_change_pre, _yurameki_cache_frame_change_pre),
        (bpy.app.handlers.frame_change_post, _yurameki_cache_frame_change),
        (bpy.app.handlers.load_pre, _yurameki_cache_load_pre),
    ):
        if function in handlers:
            handlers.remove(function)
    _CACHE_REGISTRY.clear()


def check_native_ready(curves_obj, collider_objects, root_locked_points: int,
                       particle_mass: float) -> NativeCheckStats:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    pps, strands = _uniform_points_per_strand(curves_obj)
    world, _original = _read_world(curves_obj)
    simulator = native.Simulator(world.reshape(strands, pps, 3), world.reshape(strands, pps, 3), pps,
                                 max(1, min(int(root_locked_points), pps)), float(particle_mass), 1)
    vertices, triangles = _evaluated_mesh_arrays(collider_objects)
    if not len(triangles):
        raise ValueError("Collider meshes have no evaluated triangles")
    return NativeCheckStats(
        n_strands=strands, points_per_strand=pps, n_points=len(world),
        n_segments=strands * (pps - 1),
        root_locked_points=max(1, min(int(root_locked_points), pps)),
        n_vertices=len(vertices), n_triangles=len(triangles), device="cpu:openmp",
        device_name=f"{native.max_threads()} logical processors",
        native_version=str(native.__version__),
    )


def audit_frame(curves_obj, collider_objects, frame: int | None = None,
                root_locked_points: int = 3, particle_mass: float = 0.003,
                parameters: dict | None = None) -> dict:
    """Read-only C++ quality/collision audit of one evaluated Blender frame."""
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    colliders = [obj for obj in collider_objects if obj is not None and obj.type == "MESH"]
    if not colliders:
        raise ValueError("expected at least one Mesh collider")
    scene = bpy.context.scene
    restore_frame = int(scene.frame_current)
    audit_frame_number = restore_frame if frame is None else int(frame)
    pps, strands = _uniform_points_per_strand(curves_obj)
    try:
        scene.frame_set(1)
        bpy.context.view_layer.update()
        frame1_world, _frame1_original = _read_world(curves_obj)
        scene.frame_set(audit_frame_number)
        bpy.context.view_layer.update()
        evaluated, _original = _read_world(curves_obj)
        body_vertices, body_triangles = _evaluated_mesh_arrays([colliders[0]])
        clothes_vertices, clothes_triangles = _evaluated_mesh_arrays(colliders[1:])
        if len(body_triangles) == 0:
            raise ValueError("Body collider has no evaluated triangles")
    finally:
        scene.frame_set(restore_frame)
        bpy.context.view_layer.update()
    simulator = native.Simulator(
        evaluated.reshape(strands, pps, 3), frame1_world.reshape(strands, pps, 3),
        pps, max(1, min(int(root_locked_points), pps)), float(particle_mass), 1,
    )
    stats = simulator.audit(
        evaluated.reshape(strands, pps, 3), body_vertices, body_triangles,
        clothes_vertices, clothes_triangles, parameters,
    )
    stats["frame"] = audit_frame_number
    stats["n_strands"] = strands
    stats["smooth_strands"] = int(stats["settled_strands"])
    stats["disturbed_or_colliding_strands"] = int(stats["repaired_strands"])
    return stats


def _parameter_dict(**values) -> dict:
    return values


def simulate_iter(
    curves_obj,
    collider_objects,
    start_frame: int,
    end_frame: int,
    root_locked_points: int,
    gravity: tuple[float, float, float],
    damping: float,
    max_velocity_mps: float,
    particle_mass: float,
    iterations: int,
    bend_stiffness: float,
    collision_margin_m: float,
    collision_search_m: float,
    collision_max_correction_m: float,
    collision_response: float,
    collision_velocity_damping: float,
    collision_passes: int,
    post_collision_iterations: int,
    max_move_per_substep_m: float,
    max_substeps: int,
    bake_mode: str,
    guide_decimation: int = 1,
    keep_length: bool = True,
    internal_damping: float = 0.05,
    collision_smoothing: float = 0.5,
    adaptive_root_lock: bool = True,
    settle_iterations: int = 12,
    settle_relaxation: float = 0.5,
    groom_strength: float = 0.15,
    groom_repair_strength: float = 0.4,
    length_tolerance_m: float = 0.0001,
    angle_change_limit_deg: float = 30.0,
    fold_limit_deg: float = 90.0,
    roughness_factor: float = 1.25,
    settle_stagnation: int = 3,
    collision_smooth_passes: int = 4,
    surface_feedback_iterations: int = 4,
    openmp_threads: int = 0,
) -> SimStats:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    colliders = [obj for obj in collider_objects if obj is not None and obj.type == "MESH"]
    if not colliders:
        raise ValueError("expected at least one Mesh collider")
    start_frame, end_frame = int(start_frame), int(end_frame)
    if end_frame <= start_frame:
        raise ValueError("End Frame must be > Start Frame")
    bake_mode = str(bake_mode).upper()
    if bake_mode not in {"CACHE", "FINAL", "KEYFRAMES"}:
        bake_mode = "CACHE"
    guide_decimation = max(1, int(guide_decimation))
    surface_feedback_iterations = max(1, int(surface_feedback_iterations))
    frames = list(range(start_frame, end_frame + 1))
    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    started = time.perf_counter()
    pps, strands = _uniform_points_per_strand(curves_obj)
    locked = max(1, min(int(root_locked_points), pps))

    previous_cache = get_runtime_cache(curves_obj)
    restore_local = previous_cache.restore_local_values.copy() if previous_cache is not None else _read_local_points(curves_obj)
    clear_runtime_cache(curves_obj)
    target_frames = sorted(set(frames + [1, original_frame]))

    global _CACHE_MUTED
    old_cache_muted = _CACHE_MUTED
    _CACHE_MUTED = True
    target_worlds: dict[int, np.ndarray] = {}
    target_offsets: dict[int, np.ndarray] = {}
    local_results: dict[int, np.ndarray] = {}
    cache_path = ""
    completed = False
    cancelled = False

    total_substeps = total_hits = frame_steps = 0
    max_substeps_seen = 1
    max_auto_move_mm = max_pre_groom_length_error_mm = max_final_length_error_mm = 0.0
    groomed_total = failed_total = length_bad_total = length_bad_rods_total = folded_total = 0
    shape_bad_total = rough_bad_total = collision_bad_total = 0
    unresolved_shape_total = unresolved_collision_total = 0
    adaptive_lock_max = adaptive_lock_strand_frames = 0
    max_settle_used = feedback_repairs = 0
    last_triangle_count = 0
    workers = min(int(openmp_threads), int(native.max_threads())) if int(openmp_threads) > 0 else int(native.max_threads())

    parameters = _parameter_dict(
        gravity=np.ascontiguousarray(gravity, dtype=np.float32), damping=float(damping),
        internal_damping=float(internal_damping), max_velocity=float(max_velocity_mps),
        iterations=max(1, int(iterations)), stretch_stiffness=STRETCH_STIFFNESS,
        bend_stiffness=float(bend_stiffness), collision_margin=float(collision_margin_m),
        collision_search=float(collision_search_m), collision_max_correction=float(collision_max_correction_m),
        collision_response=float(collision_response), collision_velocity_damping=float(collision_velocity_damping),
        collision_smoothing=float(collision_smoothing), collision_passes=max(1, int(collision_passes)),
        post_collision_iterations=max(0, int(post_collision_iterations)),
        max_move_per_substep=float(max_move_per_substep_m), max_substeps=max(1, int(max_substeps)),
        keep_length=bool(keep_length), adaptive_root_lock=bool(adaptive_root_lock),
        settle_iterations=max(1, int(settle_iterations)), settle_relaxation=float(settle_relaxation),
        groom_strength=float(groom_strength), repair_strength=float(groom_repair_strength),
        length_tolerance=float(length_tolerance_m), angle_change_limit_deg=float(angle_change_limit_deg),
        fold_limit_deg=float(fold_limit_deg), roughness_factor=float(roughness_factor),
        settle_stagnation=max(1, int(settle_stagnation)),
        collision_smooth_passes=max(0, int(collision_smooth_passes)), openmp_threads=max(0, int(openmp_threads)),
    )

    try:
        target_worlds, target_offsets, target_locals = _read_targets_for_frames(curves_obj, target_frames, restore_local)
        initial = target_worlds[start_frame].reshape(strands, pps, 3)
        frame1_rest = target_worlds[1].reshape(strands, pps, 3)
        simulator = native.Simulator(initial, frame1_rest, pps, locked, float(particle_mass), guide_decimation)
        body_obj = colliders[0]

        scene.frame_set(start_frame)
        _write_local_points(curves_obj, target_locals[start_frame])
        bpy.context.view_layer.update()
        simulator.prime_head(_head_frame_world(body_obj) if adaptive_root_lock else None)
        local_results[start_frame] = _read_local_points(curves_obj)
        _force_viewport_refresh()

        dt_frame = float(scene.render.fps_base) / max(float(scene.render.fps), 1.0)
        for frame in frames[1:]:
            scene.frame_set(frame)
            _write_local_points(curves_obj, target_locals[frame])
            bpy.context.view_layer.update()
            evaluated = target_worlds[frame]
            body_vertices, body_triangles = _evaluated_mesh_arrays([body_obj])
            clothes_vertices, clothes_triangles = _evaluated_mesh_arrays(colliders[1:])
            if not len(body_triangles):
                raise ValueError(f"Body collider has no evaluated triangles at frame {frame}")
            last_triangle_count = len(body_triangles) + len(clothes_triangles)
            result = simulator.frame(
                evaluated.reshape(strands, pps, 3), body_vertices, body_triangles,
                clothes_vertices, clothes_triangles, dt_frame, parameters,
                _head_frame_world(body_obj) if adaptive_root_lock else None,
            )
            main_stats = result["stats"]
            desired = np.ascontiguousarray(result["positions"].reshape(-1, 3), dtype=np.float32)
            repaired_mask = np.asarray(result["repaired_mask"], dtype=np.int32).copy()
            current_offset = target_offsets[frame]
            final_evaluated = desired
            final_stats = main_stats
            stable = False
            unresolved_this_frame = 0

            for _feedback in range(surface_feedback_iterations):
                _write_world_points(curves_obj, desired, offset=current_offset)
                bpy.context.view_layer.update()
                actual_evaluated, actual_original = _read_world(curves_obj)
                settled = simulator.settle_external(actual_evaluated.reshape(strands, pps, 3), parameters)
                settle_stats = settled["stats"]
                settle_mask = np.asarray(settled["repaired_mask"], dtype=np.int32)
                repaired_mask |= settle_mask
                final_evaluated = actual_evaluated
                final_stats = settle_stats
                if int(settle_stats["repaired_strands"]) == 0 and int(settle_stats["failed_strands"]) == 0:
                    stable = True
                    break
                feedback_repairs += int(np.count_nonzero(settle_mask))
                desired = np.ascontiguousarray(settled["positions"].reshape(-1, 3), dtype=np.float32)
                current_offset = actual_evaluated - actual_original

            if not stable:
                _write_world_points(curves_obj, desired, offset=current_offset)
                bpy.context.view_layer.update()
                final_evaluated, _final_original = _read_world(curves_obj)
                audit = simulator.settle_external(final_evaluated.reshape(strands, pps, 3), parameters)
                final_stats = audit["stats"]
                audit_mask = np.asarray(audit["repaired_mask"], dtype=np.int32)
                repaired_mask |= audit_mask
                # The hard feedback cap was reached.  The audit candidate is
                # intentionally not written again (that would evade the cap),
                # so every strand it wanted to repair remains unresolved in the
                # actually visible evaluated Curves state.
                unresolved_this_frame = int(np.count_nonzero(audit_mask))

            simulator.sync_external(final_evaluated.reshape(strands, pps, 3), repaired_mask)
            local_results[frame] = _read_local_points(curves_obj)
            frame_steps += 1
            total_substeps += int(main_stats["substeps"])
            max_substeps_seen = max(max_substeps_seen, int(main_stats["substeps"]))
            total_hits += int(main_stats["hits"])
            max_auto_move_mm = max(max_auto_move_mm, float(main_stats["auto_move_mm"]))
            max_pre_groom_length_error_mm = max(max_pre_groom_length_error_mm, float(main_stats["max_length_error_mm"]))
            max_final_length_error_mm = max(max_final_length_error_mm, float(final_stats["max_length_error_mm"]))
            groomed_total += int(np.count_nonzero(repaired_mask))
            failed_total += unresolved_this_frame
            length_bad_total += int(main_stats["length_bad_before"])
            length_bad_rods_total += int(main_stats["length_bad_rods_before"])
            folded_total += int(main_stats["folded_before"])
            shape_bad_total += int(main_stats["shape_bad_before"])
            rough_bad_total += int(main_stats["rough_bad_before"])
            collision_bad_total += int(main_stats["collision_bad_before"])
            unresolved_shape_total += int(final_stats["shape_bad_before"])
            unresolved_collision_total += int(final_stats["collision_bad_before"])
            adaptive_lock_max = max(adaptive_lock_max, int(main_stats["adaptive_lock_max_points"]))
            adaptive_lock_strand_frames += int(main_stats["adaptive_lock_strands"])
            max_settle_used = max(max_settle_used, int(main_stats["max_settle_iterations"]), int(final_stats["max_settle_iterations"]))
            print(
                f"Yurameki C++ frame={frame}/{end_frame} substeps={main_stats['substeps']} "
                f"groomed={np.count_nonzero(repaired_mask)} unresolved={unresolved_this_frame} "
                f"hits={main_stats['hits']} threads={workers}"
            )
            _force_viewport_refresh()
            yield {
                "frame": frame, "start_frame": start_frame, "end_frame": end_frame,
                "completed": frame_steps, "total": len(frames) - 1,
            }

        output_values = np.ascontiguousarray([local_results[frame] for frame in frames], dtype=np.float32)
        if bake_mode == "KEYFRAMES":
            _bake_local_position_keyframes(curves_obj, frames, output_values)
        elif bake_mode == "CACHE":
            cache = _register_sim_cache(curves_obj, frames, output_values, restore_local)
            cache_path = cache.path
        else:
            scene.frame_set(end_frame)
            _write_local_points(curves_obj, local_results[end_frame])
            bpy.context.view_layer.update()
        completed = True
    except GeneratorExit:
        cancelled = True
        computed_frames = sorted(local_results)
        if len(computed_frames) >= 2:
            values = np.ascontiguousarray([local_results[frame] for frame in computed_frames], dtype=np.float32)
            try:
                cache_path = _register_sim_cache(curves_obj, computed_frames, values, restore_local).path
            except Exception as exc:
                print(f"Yurameki could not save the partial cache: {exc!r}")
        raise
    finally:
        _CACHE_MUTED = old_cache_muted
        if not completed and not cancelled:
            _write_local_points(curves_obj, restore_local)
            scene.frame_set(original_frame)
            bpy.context.view_layer.update()

    return SimStats(
        start_frame=start_frame, end_frame=end_frame, n_frames=len(frames),
        n_strands=strands, simulated_strands=simulator.guide_count,
        guide_decimation=guide_decimation, points_per_strand=pps,
        root_locked_points=locked, adaptive_root_lock=bool(adaptive_root_lock),
        adaptive_lock_max_points=adaptive_lock_max,
        adaptive_lock_strand_frames=adaptive_lock_strand_frames,
        frame_steps=frame_steps, total_substeps=total_substeps,
        max_substeps=max_substeps_seen, max_auto_move_mm=max_auto_move_mm,
        total_hits=total_hits, n_triangles_last=last_triangle_count,
        bake_mode=bake_mode, max_velocity_mps=float(max_velocity_mps),
        collision_max_correction_mm=float(collision_max_correction_m) * 1000.0,
        collision_response=float(collision_response),
        collision_velocity_damping=float(collision_velocity_damping),
        keep_length=bool(keep_length), keep_length_source_frame=1,
        max_keep_length_error_mm=max_final_length_error_mm,
        cache_path=cache_path, device="cpu:openmp",
        device_name=f"{workers} OpenMP threads",
        elapsed_sec=time.perf_counter() - started,
        groomed_strand_frames=groomed_total,
        settle_failed_strand_frames=failed_total,
        length_bad_strand_frames=length_bad_total,
        length_bad_rod_frames=length_bad_rods_total,
        folded_strand_frames=folded_total,
        max_settle_iterations=max_settle_used,
        surface_feedback_repairs=feedback_repairs,
        shape_bad_strand_frames=shape_bad_total,
        rough_strand_frames=rough_bad_total,
        collision_bad_strand_frames=collision_bad_total,
        unresolved_shape_strand_frames=unresolved_shape_total,
        unresolved_collision_strand_frames=unresolved_collision_total,
        max_pre_groom_length_error_mm=max_pre_groom_length_error_mm,
    )


def simulate(*args, **kwargs) -> SimStats:
    generator = simulate_iter(*args, **kwargs)
    try:
        while True:
            next(generator)
    except StopIteration as stopped:
        return stopped.value
