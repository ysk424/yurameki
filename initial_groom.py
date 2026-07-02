"""CPU BVH initial groom for straight long hair.

This is intentionally a groom/setup pass, not a per-frame simulation.  It lays
strands from the root toward the back and then down, using Blender's BVHTree to
avoid the body.  Near the body it briefly slides on the tangent plane, then
releases back to vertical falling when a short downward probe is clear.
"""

from __future__ import annotations

import time

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


BACK = Vector((0.0, 1.0, 0.0))
DOWN = Vector((0.0, 0.0, -1.0))


def _curve_spans(curves_data):
    return [
        (int(curve.first_point_index), int(curve.points_length))
        for curve in curves_data.curves
    ]


def _ensure_backup(curves_obj):
    name = f"{curves_obj.name}_yurameki_backup_before_settle_hair_back"
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing.name
    data = curves_obj.data.copy()
    obj = bpy.data.objects.new(name, data)
    obj.matrix_world = curves_obj.matrix_world.copy()
    curves_obj.users_collection[0].objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    return obj.name


def _body_bvh(collider_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = collider_obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    try:
        vertices = [obj_eval.matrix_world @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(poly.vertices) for poly in mesh.polygons]
        return BVHTree.FromPolygons(vertices, polygons)
    finally:
        obj_eval.to_mesh_clear()


def _base_drop_direction(t: float) -> Vector:
    rear = max(0.0, 1.0 - t / 0.28)
    down_w = 0.35 + 4.5 * (t ** 0.55)
    back_w = 1.35 * rear + 0.03
    direction = BACK * back_w + DOWN * down_w
    if direction.length <= 1.0e-9:
        direction = DOWN.copy()
    direction.normalize()
    return direction


def _project_to_tangent(direction: Vector, normal: Vector) -> Vector:
    slide = direction - normal * direction.dot(normal)
    if slide.length <= 1.0e-7:
        slide = BACK - normal * BACK.dot(normal)
    if slide.length <= 1.0e-7:
        slide = DOWN.copy()
    slide.normalize()
    return slide


def settle_hair_back(
    curves_obj,
    collider_obj,
    max_strands: int = 500,
    collision_radius_m: float = 0.0025,
    follow_radius_m: float = 0.0300,
    release_probe_m: float = 0.0200,
    release_clearance_m: float = 0.0040,
    max_surface_run_m: float = 0.0300,
    surface_stick: float = 0.78,
    push_iterations: int = 5,
) -> dict:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if collider_obj is None or collider_obj.type != "MESH":
        raise ValueError("expected one Mesh collider")

    start_time = time.perf_counter()
    backup_name = _ensure_backup(curves_obj)
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    spans = _curve_spans(curves_obj.data)
    if not spans:
        raise ValueError("Curves object has no strands")

    n_total = len(attr.data)
    flat = [0.0] * (n_total * 3)
    attr.data.foreach_get("vector", flat)
    world_m = curves_obj.matrix_world
    world_inv = curves_obj.matrix_world.inverted()
    world_pts = [
        world_m @ Vector((flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]))
        for i in range(n_total)
    ]
    bvh = _body_bvh(collider_obj)

    root_entries = [(world_pts[start].z, si) for si, (start, _count) in enumerate(spans)]
    root_entries.sort(key=lambda item: (item[0], item[1]))
    if max_strands <= 0:
        target_strands = [si for _z, si in root_entries]
    else:
        target_strands = [si for _z, si in root_entries[:max_strands]]

    stats = {
        "slide_events": 0,
        "release_events": 0,
        "forced_release_events": 0,
        "ray_hits": 0,
        "nearest_pushes": 0,
        "sample_pushes": 0,
        "remaining_close_points": 0,
        "changed_points": 0,
        "max_move_m": 0.0,
        "max_len_error_m": 0.0,
        "min_clearance_m": 999.0,
        "tip_down_dot_sum": 0.0,
    }

    def clear_along(point: Vector, direction: Vector, distance: float, clearance: float) -> bool:
        hit = bvh.ray_cast(point, direction, distance)
        if hit is not None:
            loc, _normal, _index, dist = hit
            if loc is not None and dist is not None and 0.0002 < dist <= distance:
                return False
        end = point + direction * distance
        nearest = bvh.find_nearest(end, max(clearance * 3.0, follow_radius_m))
        if nearest is None:
            return True
        _loc, _normal, _index, dist = nearest
        return dist is None or dist >= clearance

    def choose_direction(point: Vector, desired_dir: Vector, surface_run: float):
        nearest = bvh.find_nearest(point, follow_radius_m)
        if nearest is None:
            return desired_dir, 0.0, False
        _loc, normal, _index, dist = nearest
        if normal is None or dist is None:
            return desired_dir, 0.0, False
        stats["min_clearance_m"] = min(stats["min_clearance_m"], float(dist))
        normal = normal.normalized()
        if dist >= follow_radius_m:
            return desired_dir, 0.0, False

        safe_down = clear_along(point, DOWN, release_probe_m, release_clearance_m)
        down_not_into_body = DOWN.dot(normal) > -0.25
        if safe_down and (down_not_into_body or surface_run > collision_radius_m):
            stats["release_events"] += 1
            return DOWN.copy(), 0.0, False

        slide = _project_to_tangent(desired_dir, normal)
        weight = max(0.0, min(1.0, (follow_radius_m - dist) / follow_radius_m)) * surface_stick
        mixed = desired_dir * (1.0 - weight) + slide * weight
        if surface_run >= max_surface_run_m:
            outward_down = DOWN * 0.78 + normal * 0.35
            if outward_down.length <= 1.0e-7:
                outward_down = DOWN.copy()
            outward_down.normalize()
            mixed = mixed * 0.35 + outward_down * 0.65
            stats["forced_release_events"] += 1
        if mixed.length <= 1.0e-7:
            mixed = slide
        mixed.normalize()
        stats["slide_events"] += 1
        return mixed, surface_run + 0.01, True

    def solve_candidate(prev: Vector, desired: Vector, seg_len: float) -> Vector:
        candidate = desired.copy()
        fallback_dir = candidate - prev
        if fallback_dir.length <= 1.0e-9:
            fallback_dir = DOWN.copy()
        fallback_dir.normalize()
        for _it in range(push_iterations):
            move = candidate - prev
            move_len = move.length
            if move_len > 1.0e-9:
                direction = move.normalized()
                hit = bvh.ray_cast(prev, direction, move_len + collision_radius_m)
                if hit is not None:
                    loc, normal, _index, dist = hit
                    if (
                        loc is not None
                        and normal is not None
                        and dist is not None
                        and 0.0002 < dist <= move_len + collision_radius_m
                    ):
                        normal = normal.normalized()
                        slide = _project_to_tangent(direction, normal)
                        candidate = loc + normal * collision_radius_m + slide * max(0.0, seg_len - dist)
                        v = candidate - prev
                        candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                        stats["ray_hits"] += 1

            nearest = bvh.find_nearest(candidate, follow_radius_m)
            if nearest is not None:
                loc, normal, _index, dist = nearest
                if loc is not None and normal is not None and dist is not None:
                    stats["min_clearance_m"] = min(stats["min_clearance_m"], float(dist))
                    if dist < collision_radius_m:
                        normal = normal.normalized()
                        tangent = _project_to_tangent(candidate - prev, normal)
                        candidate = loc + normal * collision_radius_m + tangent * 0.0005
                        v = candidate - prev
                        candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                        stats["nearest_pushes"] += 1

            mid = prev.lerp(candidate, 0.5)
            nearest_mid = bvh.find_nearest(mid, follow_radius_m)
            if nearest_mid is not None:
                _loc, normal, _index, dist = nearest_mid
                if normal is not None and dist is not None:
                    stats["min_clearance_m"] = min(stats["min_clearance_m"], float(dist))
                    if dist < collision_radius_m:
                        candidate = candidate + normal.normalized() * (collision_radius_m - dist)
                        v = candidate - prev
                        candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                        stats["sample_pushes"] += 1
        v = candidate - prev
        if v.length > 1.0e-9:
            return prev + v.normalized() * seg_len
        return prev + fallback_dir * seg_len

    for si in target_strands:
        start, count = spans[si]
        old = [world_pts[start + j].copy() for j in range(count)]
        seg_lens = []
        for j in range(count - 1):
            length = (old[j + 1] - old[j]).length
            seg_lens.append(length if length > 1.0e-6 else 0.01)

        new = [old[0].copy()]
        surface_run = 0.0
        for j, seg_len in enumerate(seg_lens):
            t = j / max(1, len(seg_lens) - 1)
            desired_dir = _base_drop_direction(t)
            direction, surface_run, in_surface = choose_direction(new[-1], desired_dir, surface_run)
            if not in_surface:
                surface_run = 0.0
            new.append(solve_candidate(new[-1], new[-1] + direction * seg_len, seg_len))

        for _pass in range(3):
            surface_run = 0.0
            for j, seg_len in enumerate(seg_lens):
                direction = new[j + 1] - new[j]
                if direction.length <= 1.0e-9:
                    direction = _base_drop_direction(j / max(1, len(seg_lens) - 1))
                else:
                    direction.normalize()
                direction, surface_run, in_surface = choose_direction(new[j], direction, surface_run)
                if not in_surface:
                    surface_run = 0.0
                new[j + 1] = solve_candidate(new[j], new[j] + direction * seg_len, seg_len)

        tip_dir = new[-1] - new[-2]
        if tip_dir.length > 1.0e-9:
            stats["tip_down_dot_sum"] += tip_dir.normalized().dot(DOWN)

        for j in range(count):
            idx = start + j
            stats["max_move_m"] = max(stats["max_move_m"], (new[j] - old[j]).length)
            world_pts[idx] = new[j]
            stats["changed_points"] += 1
            nearest = bvh.find_nearest(new[j], follow_radius_m)
            if nearest is not None:
                _loc, _normal, _index, dist = nearest
                if dist is not None:
                    stats["min_clearance_m"] = min(stats["min_clearance_m"], float(dist))
                    if dist < collision_radius_m * 0.98:
                        stats["remaining_close_points"] += 1
        for j, seg_len in enumerate(seg_lens):
            stats["max_len_error_m"] = max(
                stats["max_len_error_m"],
                abs((new[j + 1] - new[j]).length - seg_len),
            )

    out = []
    for point in world_pts:
        local = world_inv @ point
        out.extend((local.x, local.y, local.z))
    attr.data.foreach_set("vector", out)
    curves_obj.data.update_tag()
    bpy.context.view_layer.update()

    processed = len(target_strands)
    return {
        "backup_object": backup_name,
        "processed_strands": processed,
        "changed_points": int(stats["changed_points"]),
        "elapsed_sec": float(time.perf_counter() - start_time),
        "slide_events": int(stats["slide_events"]),
        "release_events": int(stats["release_events"]),
        "forced_release_events": int(stats["forced_release_events"]),
        "ray_hits": int(stats["ray_hits"]),
        "nearest_pushes": int(stats["nearest_pushes"]),
        "sample_pushes": int(stats["sample_pushes"]),
        "remaining_close_points": int(stats["remaining_close_points"]),
        "min_clearance_mm": None if stats["min_clearance_m"] == 999.0 else stats["min_clearance_m"] * 1000.0,
        "max_move_cm": stats["max_move_m"] * 100.0,
        "max_length_error_mm": stats["max_len_error_m"] * 1000.0,
        "avg_tip_down_dot": stats["tip_down_dot_sum"] / max(1, processed),
    }
