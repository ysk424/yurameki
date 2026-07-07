"""Collider proxy helpers for Yurameki.

The proxy keeps the source object's modifiers, but uses a private mesh copy with
boundary holes capped so parity checks can treat the body as closed.
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


def _ordered_boundary_loop(start_edge, remaining_edges: set) -> list:
    if start_edge not in remaining_edges:
        return []
    remaining_edges.remove(start_edge)
    start_vert = start_edge.verts[0]
    current_vert = start_edge.verts[1]
    previous_edge = start_edge
    verts = [start_vert, current_vert]

    while current_vert is not start_vert:
        next_edge = None
        for edge in current_vert.link_edges:
            if edge is previous_edge or edge not in remaining_edges or not edge.is_boundary:
                continue
            next_edge = edge
            break
        if next_edge is None:
            break
        remaining_edges.remove(next_edge)
        next_vert = next_edge.verts[1] if next_edge.verts[0] is current_vert else next_edge.verts[0]
        if next_vert is start_vert:
            break
        verts.append(next_vert)
        previous_edge = next_edge
        current_vert = next_vert
    return verts


def _cap_boundary_loops(mesh) -> tuple[int, int, int, int]:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        verts_before = len(bm.verts)
        faces_before = len(bm.faces)
        boundary_edges = [edge for edge in bm.edges if edge.is_boundary]
        boundary_before = len(boundary_edges)
        loops_filled = 0
        remaining = set(boundary_edges)
        while remaining:
            loop = _ordered_boundary_loop(next(iter(remaining)), remaining)
            if len(loop) < 3:
                continue
            center = sum((vert.co for vert in loop), loop[0].co.copy() * 0.0) / float(len(loop))
            center_vert = bm.verts.new(center)
            loops_filled += 1
            for index, vert in enumerate(loop):
                next_vert = loop[(index + 1) % len(loop)]
                try:
                    bm.faces.new((vert, next_vert, center_vert))
                except ValueError:
                    pass
        if loops_filled:
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            bm.normal_update()
            bm.to_mesh(mesh)
            mesh.update()
        faces_after = len(bm.faces)
        verts_after = len(bm.verts)
        boundary_after = sum(1 for edge in bm.edges if edge.is_boundary)
        return boundary_before, boundary_after, faces_after - faces_before, verts_after - verts_before
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


def _hide_proxy_for_viewport(proxy_obj) -> None:
    proxy_obj.hide_render = True
    proxy_obj.hide_select = True
    proxy_obj.display_type = "WIRE"
    proxy_obj.show_in_front = False
    try:
        proxy_obj.hide_set(True)
    except Exception:
        pass


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
        _hide_proxy_for_viewport(proxy)
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

    collections = tuple(source_obj.users_collection)
    if collections:
        collections[0].objects.link(proxy)
    else:
        bpy.context.scene.collection.objects.link(proxy)
    _hide_proxy_for_viewport(proxy)

    ear_faces_removed = _remove_ear_protrusions(proxy)
    boundary_before, boundary_after, faces_added, cap_vertices_added = _cap_boundary_loops(proxy.data)
    bpy.context.view_layer.update()

    return {
        "proxy_name": proxy.name,
        "source_name": source_obj.name,
        "boundary_edges_before": int(boundary_before),
        "boundary_edges_after": int(boundary_after),
        "faces_added": int(faces_added),
        "cap_vertices_added": int(cap_vertices_added),
        "ear_faces_removed": int(ear_faces_removed),
        "modifiers": len(proxy.modifiers),
    }
