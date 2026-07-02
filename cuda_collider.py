"""ctypes bridge for the Yurameki native CUDA collider DLL."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import bpy
import numpy as np

from . import solver_interface as si


DLL_NAME = "yurameki_cuda_collide.dll"


@dataclass
class ColliderDetectionResult:
    n_cylinders: int
    n_triangles: int
    hit_count: int
    hit_flags: np.ndarray
    hit_triangles: np.ndarray
    hit_distances: np.ndarray
    hit_normals: np.ndarray


@dataclass
class ColliderAvoidanceResult:
    n_cylinders: int
    n_triangles: int
    hit_count: int
    hit_flags: np.ndarray
    hit_triangles: np.ndarray
    hit_distances: np.ndarray
    hit_normals: np.ndarray
    max_length_error_mm: float
    max_tip_adjust_mm: float


def _dll_candidates():
    base = os.path.dirname(__file__)
    return [
        os.path.join(base, "native", DLL_NAME),
        os.path.join(base, DLL_NAME),
    ]


def _load_dll():
    path = next((candidate for candidate in _dll_candidates() if os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError(
            f"{DLL_NAME} was not found. Build native/build.ps1 first. "
            f"Searched: {', '.join(_dll_candidates())}"
        )
    dll = ctypes.CDLL(path)
    dll.yurameki_cuda_detect_capsule_mesh.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.yurameki_cuda_detect_capsule_mesh.restype = ctypes.c_int
    dll.yurameki_cuda_avoid_capsule_mesh.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.yurameki_cuda_avoid_capsule_mesh.restype = ctypes.c_int
    dll.yurameki_cuda_last_error.argtypes = []
    dll.yurameki_cuda_last_error.restype = ctypes.c_char_p
    return dll


def _as_float_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _as_int_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def mesh_world_triangles(mesh_obj):
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise ValueError("collider object must be a Mesh")
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = mesh_obj.evaluated_get(dg)
    mesh = obj_eval.to_mesh()
    try:
        mesh.calc_loop_triangles()
        mw = np.array(obj_eval.matrix_world, dtype=np.float32)
        vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        for i, vertex in enumerate(mesh.vertices):
            co = np.array((vertex.co.x, vertex.co.y, vertex.co.z, 1.0), dtype=np.float32)
            vertices[i] = (mw @ co)[:3]
        triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        for i, tri in enumerate(mesh.loop_triangles):
            triangles[i] = tri.vertices
    finally:
        obj_eval.to_mesh_clear()
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("collider mesh has no triangles")
    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(triangles, dtype=np.int32),
    )


def detect_capsule_mesh(curves_obj, collider_obj, points_per_strand: int,
                        radius_m: float, sort_axis: str = "Z",
                        cylinder_length_m: float = si.DEFAULT_CYLINDER_LENGTH
                        ) -> ColliderDetectionResult:
    dll = _load_dll()
    model = si.build_cylinder_model(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        cylinder_length_m=cylinder_length_m,
    )
    vertices, triangles = mesh_world_triangles(collider_obj)
    roots = np.ascontiguousarray(model.roots, dtype=np.float32)
    tips = np.ascontiguousarray(
        model.roots + model.dirs * model.lengths[:, None],
        dtype=np.float32,
    )
    n = int(model.n_cylinders)
    hit_flags = np.zeros(n, dtype=np.int32)
    hit_triangles = np.full(n, -1, dtype=np.int32)
    hit_distances = np.full(n, np.inf, dtype=np.float32)
    hit_normals = np.zeros((n, 3), dtype=np.float32)
    hit_count = np.zeros(1, dtype=np.int32)

    code = dll.yurameki_cuda_detect_capsule_mesh(
        n,
        _as_float_ptr(roots),
        _as_float_ptr(tips),
        ctypes.c_float(float(radius_m)),
        int(len(vertices)),
        _as_float_ptr(vertices),
        int(len(triangles)),
        _as_int_ptr(triangles),
        _as_int_ptr(hit_flags),
        _as_int_ptr(hit_triangles),
        _as_float_ptr(hit_distances),
        _as_float_ptr(hit_normals),
        _as_int_ptr(hit_count),
    )
    if code != 0:
        message = dll.yurameki_cuda_last_error()
        raise RuntimeError(message.decode("utf-8", errors="replace") if message else code)

    return ColliderDetectionResult(
        n_cylinders=n,
        n_triangles=int(len(triangles)),
        hit_count=int(hit_count[0]),
        hit_flags=hit_flags,
        hit_triangles=hit_triangles,
        hit_distances=hit_distances,
        hit_normals=hit_normals,
    )


def apply_capsule_mesh_avoidance(curves_obj, collider_obj, points_per_strand: int,
                                 radius_m: float, sort_axis: str = "Z",
                                 cylinder_length_m: float = si.DEFAULT_CYLINDER_LENGTH,
                                 n_substeps: int = 8,
                                 max_move_m: float = 0.001
                                 ) -> ColliderAvoidanceResult:
    dll = _load_dll()
    model = si.build_cylinder_model(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        cylinder_length_m=cylinder_length_m,
    )
    vertices, triangles = mesh_world_triangles(collider_obj)
    roots = np.ascontiguousarray(model.roots, dtype=np.float32)
    tips = np.ascontiguousarray(
        model.roots + model.dirs * model.lengths[:, None],
        dtype=np.float32,
    )
    lengths = np.ascontiguousarray(model.lengths, dtype=np.float32)
    n = int(model.n_cylinders)
    adjusted_tips = np.zeros_like(tips)
    hit_flags = np.zeros(n, dtype=np.int32)
    hit_triangles = np.full(n, -1, dtype=np.int32)
    hit_distances = np.full(n, np.inf, dtype=np.float32)
    hit_normals = np.zeros((n, 3), dtype=np.float32)
    hit_count = np.zeros(1, dtype=np.int32)

    code = dll.yurameki_cuda_avoid_capsule_mesh(
        n,
        _as_float_ptr(roots),
        _as_float_ptr(tips),
        _as_float_ptr(lengths),
        ctypes.c_float(float(radius_m)),
        ctypes.c_int(max(1, int(n_substeps))),
        ctypes.c_float(float(max_move_m)),
        int(len(vertices)),
        _as_float_ptr(vertices),
        int(len(triangles)),
        _as_int_ptr(triangles),
        _as_float_ptr(adjusted_tips),
        _as_int_ptr(hit_flags),
        _as_int_ptr(hit_triangles),
        _as_float_ptr(hit_distances),
        _as_float_ptr(hit_normals),
        _as_int_ptr(hit_count),
    )
    if code != 0:
        message = dll.yurameki_cuda_last_error()
        raise RuntimeError(message.decode("utf-8", errors="replace") if message else code)

    _eval_world, original_world = si._read_world(curves_obj)
    offset = model.world - original_world
    reconstructed = si.reconstruct_points(model, roots, adjusted_tips)
    si._write_world_points(curves_obj, reconstructed, offset=offset)

    length_error = np.linalg.norm(adjusted_tips - roots, axis=1) - lengths
    tip_adjust = np.linalg.norm(adjusted_tips - tips, axis=1)
    return ColliderAvoidanceResult(
        n_cylinders=n,
        n_triangles=int(len(triangles)),
        hit_count=int(hit_count[0]),
        hit_flags=hit_flags,
        hit_triangles=hit_triangles,
        hit_distances=hit_distances,
        hit_normals=hit_normals,
        max_length_error_mm=float(np.max(np.abs(length_error)) * 1000.0),
        max_tip_adjust_mm=float(np.max(tip_adjust) * 1000.0),
    )
