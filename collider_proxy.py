"""Collider proxy helpers for Yurameki.

The proxy keeps the source object's modifiers, but uses a private mesh copy with
boundary holes filled so parity checks can treat the body as closed.
"""

from __future__ import annotations

import bmesh
import bpy


PROXY_FLAG = "yurameki_collider_proxy"
PROXY_SOURCE = "yurameki_collider_proxy_source"

EAR_CUT_Z_MIN = 1.50
EAR_CUT_Z_MAX = 1.72
EAR_CUT_ABS_X = 0.09


def _mesh_boundary_count(mesh) -> int:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        return sum(1 for edge in bm.edges if edge.is_boundary)
    finally:
        bm.free()


def _fill_boundary_holes(mesh) -> tuple[int, int, int]:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        faces_before = len(bm.faces)
        boundary_edges = [edge for edge in bm.edges if edge.is_boundary]
        boundary_before = len(boundary_edges)
        if boundary_edges:
            bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
            bm.normal_update()
            bm.to_mesh(mesh)
            mesh.update()
        faces_after = len(bm.faces)
        boundary_after = sum(1 for edge in bm.edges if edge.is_boundary)
        return boundary_before, boundary_after, faces_after - faces_before
    finally:
        bm.free()


def _remove_ear_protrusions(proxy_obj) -> int:
    mesh = proxy_obj.data
    world = proxy_obj.matrix_world.copy()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        remove_faces = []
        for face in bm.faces:
            center = world @ face.calc_center_median()
            if (
                EAR_CUT_Z_MIN <= center.z <= EAR_CUT_Z_MAX
                and abs(center.x) >= EAR_CUT_ABS_X
            ):
                remove_faces.append(face)
        if remove_faces:
            bmesh.ops.delete(bm, geom=remove_faces, context="FACES")
            bm.normal_update()
            bm.to_mesh(mesh)
            mesh.update()
        return len(remove_faces)
    finally:
        bm.free()


def _remove_proxy_object(obj) -> None:
    mesh = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def clear_proxy(proxy_name: str) -> None:
    proxy = bpy.data.objects.get(proxy_name.strip()) if proxy_name else None
    if proxy is not None and bool(proxy.get(PROXY_FLAG, False)):
        _remove_proxy_object(proxy)


def get_valid_proxy(source_obj, proxy_name: str):
    proxy = bpy.data.objects.get(proxy_name.strip()) if proxy_name else None
    if (
        proxy is not None
        and proxy.type == "MESH"
        and bool(proxy.get(PROXY_FLAG, False))
        and proxy.get(PROXY_SOURCE) == source_obj.name
    ):
        return proxy
    return None


def build_filled_proxy(source_obj, existing_proxy_name: str = "") -> dict:
    if source_obj is None or source_obj.type != "MESH":
        raise ValueError("expected one Mesh collider")

    clear_proxy(existing_proxy_name)
    for obj in list(bpy.data.objects):
        if (
            obj is not None
            and obj.type == "MESH"
            and bool(obj.get(PROXY_FLAG, False))
            and obj.get(PROXY_SOURCE) == source_obj.name
        ):
            _remove_proxy_object(obj)

    proxy = source_obj.copy()
    proxy.data = source_obj.data.copy()
    proxy.name = f"{source_obj.name}_yurameki_proxy"
    proxy.data.name = f"{proxy.name}_mesh"
    proxy[PROXY_FLAG] = True
    proxy[PROXY_SOURCE] = source_obj.name
    proxy.hide_render = True
    proxy.display_type = "WIRE"
    proxy.show_in_front = True

    collections = tuple(source_obj.users_collection)
    if collections:
        collections[0].objects.link(proxy)
    else:
        bpy.context.scene.collection.objects.link(proxy)

    ear_faces_removed = _remove_ear_protrusions(proxy)
    boundary_before, boundary_after, faces_added = _fill_boundary_holes(proxy.data)
    bpy.context.view_layer.update()

    return {
        "proxy_name": proxy.name,
        "source_name": source_obj.name,
        "boundary_edges_before": int(boundary_before),
        "boundary_edges_after": int(boundary_after),
        "faces_added": int(faces_added),
        "ear_faces_removed": int(ear_faces_removed),
        "modifiers": len(proxy.modifiers),
    }
