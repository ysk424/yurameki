"""CPU BVH initial groom for straight long hair.

This is intentionally a groom/setup pass, not a per-frame simulation.  It lays
strands from the root toward the back and then down, using Blender's BVHTree to
avoid the body.  Near the body it briefly slides on the tangent plane, then
releases back to vertical falling when a short downward probe is clear.
"""

from __future__ import annotations

import math
import time

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


BACK = Vector((0.0, 1.0, 0.0))
DOWN = Vector((0.0, 0.0, -1.0))
LOWER_FREE_GROOM_Z = 1.42


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
    if isinstance(collider_obj, (list, tuple)):
        vertices = []
        polygons = []
        offset = 0
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in collider_obj:
            if obj is None or obj.type != "MESH":
                continue
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            try:
                vertices.extend(obj_eval.matrix_world @ vertex.co for vertex in mesh.vertices)
                polygons.extend(tuple(int(index) + offset for index in poly.vertices) for poly in mesh.polygons)
                offset += len(mesh.vertices)
            finally:
                obj_eval.to_mesh_clear()
        if not vertices or not polygons:
            raise ValueError("collider list has no mesh polygons")
        return BVHTree.FromPolygons(vertices, polygons)
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


def _limit_turn_direction(prev_dir: Vector | None, desired_dir: Vector, max_angle_rad: float) -> tuple[Vector, bool]:
    if prev_dir is None or prev_dir.length <= 1.0e-9 or desired_dir.length <= 1.0e-9:
        direction = desired_dir.copy()
        if direction.length <= 1.0e-9:
            direction = DOWN.copy()
        direction.normalize()
        return direction, False

    prev = prev_dir.normalized()
    desired = desired_dir.normalized()
    dot = max(-1.0, min(1.0, prev.dot(desired)))
    min_dot = math.cos(max_angle_rad)
    if dot >= min_dot:
        return desired, False

    tangent = desired - prev * dot
    if tangent.length <= 1.0e-7:
        tangent = BACK - prev * prev.dot(BACK)
    if tangent.length <= 1.0e-7:
        tangent = DOWN - prev * prev.dot(DOWN)
    if tangent.length <= 1.0e-7:
        tangent = Vector((1.0, 0.0, 0.0)) - prev * prev.x
    if tangent.length <= 1.0e-7:
        return prev.copy(), True
    tangent.normalize()
    limited = prev * math.cos(max_angle_rad) + tangent * math.sin(max_angle_rad)
    limited.normalize()
    return limited, True


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
    max_turn_angle_rad: float = 1.0,
) -> dict:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if isinstance(collider_obj, (list, tuple)):
        if not any(obj is not None and obj.type == "MESH" for obj in collider_obj):
            raise ValueError("expected at least one Mesh collider")
    elif collider_obj is None or collider_obj.type != "MESH":
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
    root_outward_dirs: list[Vector | None] = []
    for start, count in spans:
        if count < 2:
            root_outward_dirs.append(None)
            continue
        direction = world_pts[start + 1] - world_pts[start]
        if direction.length <= 1.0e-9:
            root_outward_dirs.append(None)
            continue
        direction.normalize()
        root_outward_dirs.append(direction)

    bvh = _body_bvh(collider_obj)
    collider_objs = list(collider_obj) if isinstance(collider_obj, (list, tuple)) else [collider_obj]
    bbox_world = [
        obj.matrix_world @ Vector(corner)
        for obj in collider_objs
        if obj is not None and obj.type == "MESH"
        for corner in obj.bound_box
    ]
    bbox_min = Vector((
        min(point.x for point in bbox_world),
        min(point.y for point in bbox_world),
        min(point.z for point in bbox_world),
    ))
    bbox_max = Vector((
        max(point.x for point in bbox_world),
        max(point.y for point in bbox_world),
        max(point.z for point in bbox_world),
    ))
    bbox_center = (bbox_min + bbox_max) * 0.5
    head_xy_extent = max(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y)
    head_push_center = Vector((bbox_center.x, bbox_center.y, bbox_center.z - head_xy_extent))
    head_region_min_z = head_push_center.z

    root_entries = [(world_pts[start].z, si) for si, (start, _count) in enumerate(spans)]
    root_entries.sort(key=lambda item: (item[0], item[1]))
    if max_strands <= 0:
        target_strands = [si for _z, si in root_entries]
    else:
        target_strands = [si for _z, si in root_entries[:max_strands]]

    stats = {
        "slide_events": 0,
        "release_events": 0,
        "back_release_events": 0,
        "release_blocked_inside": 0,
        "release_blocked_ray": 0,
        "forced_release_events": 0,
        "ray_hits": 0,
        "nearest_pushes": 0,
        "inside_pushes": 0,
        "head_radial_pushes": 0,
        "sample_pushes": 0,
        "final_guard_pushes": 0,
        "final_guard_fallbacks": 0,
        "final_guard_unresolved": 0,
        "root_emergence_guides": 0,
        "normal_root_locks": 0,
        "angle_limited_rods": 0,
        "lower_free_rods": 0,
        "remaining_close_points": 0,
        "changed_points": 0,
        "max_move_m": 0.0,
        "max_len_error_m": 0.0,
        "min_clearance_m": 999.0,
        "tip_down_dot_sum": 0.0,
    }

    back_down = (BACK * 0.55 + DOWN * 0.83).normalized()

    def signed_outside_distance(point: Vector, search_radius: float = 0.080) -> float:
        nearest = bvh.find_nearest(point, search_radius)
        if nearest is None:
            return 999.0
        loc, normal, _index, _dist = nearest
        if loc is None or normal is None:
            return 999.0
        return float((point - loc).dot(normal.normalized()))

    def release_path_outside_enough(point: Vector, direction: Vector) -> bool:
        # A single endpoint can be outside while the path still cuts behind an
        # ear/scalp feature.  Check the whole short release path instead.
        # Release requires visual clearance, not just non-penetration. Otherwise
        # a shallow outside path can look like it cuts through the skin.
        min_clearance = max(0.0, float(release_clearance_m))
        for factor in (0.25, 0.5, 0.75, 1.0):
            sample = point + direction * (release_probe_m * factor)
            if signed_outside_distance(sample) < min_clearance:
                return False
        return True

    def ray_clear(point: Vector, direction: Vector, distance: float) -> bool:
        hit = bvh.ray_cast(point, direction, distance)
        if hit is not None:
            loc, _normal, _index, dist = hit
            if loc is not None and dist is not None and 0.0002 < dist <= distance:
                return False
        return True

    def point_inside_collider(point: Vector) -> bool:
        # Count intersections in several directions. A majority vote avoids
        # edge/vertex grazing cases that make a single parity ray unreliable.
        max_span = max(
            bbox_max.x - bbox_min.x,
            bbox_max.y - bbox_min.y,
            bbox_max.z - bbox_min.z,
            0.1,
        )
        ray_distance = max_span * 3.0
        votes = 0
        for direction in (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        ):
            origin = point + direction * 1.0e-5
            count = 0
            travelled = 0.0
            while travelled < ray_distance:
                hit = bvh.ray_cast(origin, direction, ray_distance - travelled)
                if hit is None:
                    break
                loc, _normal, _index, dist = hit
                if loc is None or dist is None:
                    break
                step = max(float(dist), 1.0e-5)
                travelled += step
                count += 1
                origin = loc + direction * 1.0e-5
                travelled += 1.0e-5
            if count % 2 == 1:
                votes += 1
        return votes >= 2

    def head_radial_direction(point: Vector) -> Vector:
        direction = point - head_push_center
        direction.z = max(direction.z, 0.0)
        if direction.length <= 1.0e-7:
            direction = point - bbox_center
        if direction.length <= 1.0e-7:
            direction = BACK.copy()
        direction.normalize()
        return direction

    def outward_head_normal(point: Vector, normal: Vector, guide_dir: Vector | None = None) -> Vector:
        outward = normal.normalized()
        guide = guide_dir if guide_dir is not None and guide_dir.length > 1.0e-7 else head_radial_direction(point)
        if outward.dot(guide) < 0.0:
            outward.negate()
        return outward

    def push_direction(point: Vector, normal: Vector | None) -> Vector:
        if point.z >= head_region_min_z:
            stats["head_radial_pushes"] += 1
            return head_radial_direction(point)
        if normal is not None and normal.length > 1.0e-7:
            return normal.normalized()
        direction = point - bbox_center
        if direction.length <= 1.0e-7:
            direction = BACK.copy()
        direction.normalize()
        return direction

    def pushed_out_point(point: Vector, normal: Vector | None, min_push: float) -> Vector:
        push_dir = push_direction(point, normal)
        max_span = max(
            bbox_max.x - bbox_min.x,
            bbox_max.y - bbox_min.y,
            bbox_max.z - bbox_min.z,
            0.1,
        )
        hit = bvh.ray_cast(point + push_dir * 1.0e-5, push_dir, max_span * 3.0)
        if hit is not None:
            loc, _normal, _index, dist = hit
            if loc is not None and dist is not None and dist >= 0.0:
                return loc + push_dir * collision_radius_m
        return point + push_dir * max(min_push, collision_radius_m)

    def point_in_lower_free_region(point: Vector) -> bool:
        return point.z < LOWER_FREE_GROOM_Z

    def release_direction_if_safe(point: Vector):
        down_ray = ray_clear(point, DOWN, release_probe_m)
        down_outside = release_path_outside_enough(point, DOWN)
        if down_ray and down_outside:
            stats["release_events"] += 1
            return DOWN.copy(), True
        if not down_ray:
            stats["release_blocked_ray"] += 1
        if not down_outside:
            stats["release_blocked_inside"] += 1

        back_ray = ray_clear(point, back_down, release_probe_m)
        back_outside = release_path_outside_enough(point, back_down)
        if back_ray and back_outside:
            stats["back_release_events"] += 1
            return back_down.copy(), True
        if not back_ray:
            stats["release_blocked_ray"] += 1
        if not back_outside:
            stats["release_blocked_inside"] += 1
        return None, False

    def clear_along(point: Vector, direction: Vector, distance: float, clearance: float) -> bool:
        if not ray_clear(point, direction, distance):
            return False
        end = point + direction * distance
        nearest = bvh.find_nearest(end, max(clearance * 3.0, follow_radius_m))
        if nearest is None:
            return True
        _loc, _normal, _index, dist = nearest
        return dist is None or dist >= clearance

    def choose_direction(point: Vector, desired_dir: Vector, surface_run: float, lower_free: bool = False):
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

        if not lower_free:
            release_dir, released = release_direction_if_safe(point)
            if released:
                return release_dir, 0.0, False

        slide = _project_to_tangent(desired_dir, normal)
        weight = max(0.0, min(1.0, (follow_radius_m - dist) / follow_radius_m)) * surface_stick
        mixed = desired_dir * (1.0 - weight) + slide * weight
        if surface_run >= max_surface_run_m and not lower_free:
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

    def solve_candidate(prev: Vector, desired: Vector, seg_len: float, shallow_guard: bool = False) -> Vector:
        candidate = desired.copy()
        fallback_dir = candidate - prev
        if fallback_dir.length <= 1.0e-9:
            fallback_dir = DOWN.copy()
        fallback_dir.normalize()

        def project_to_length(point: Vector) -> Vector:
            v = point - prev
            return prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len

        def nearest_normal(point: Vector) -> Vector | None:
            nearest = bvh.find_nearest(point, follow_radius_m)
            if nearest is None:
                return None
            _loc, normal, _index, _dist = nearest
            if normal is None or normal.length <= 1.0e-7:
                return None
            return normal.normalized()

        def final_segment_issue(point: Vector):
            if point_inside_collider(point):
                return "endpoint", point, nearest_normal(point)

            move = point - prev
            move_len = move.length
            if move_len > 1.0e-9:
                direction = move.normalized()
                origin = prev + direction * 1.0e-5
                cast_distance = move_len - 2.0e-5
                if cast_distance > 1.0e-9:
                    hit = bvh.ray_cast(origin, direction, cast_distance)
                    if hit is not None:
                        loc, normal, _index, dist = hit
                        if loc is not None and normal is not None and dist is not None and dist >= 0.0:
                            return "ray", loc, normal.normalized()

            sample_factors = (0.05, 0.10, 0.20, 0.25, 0.5, 0.75) if shallow_guard else (0.25, 0.5, 0.75)
            for factor in sample_factors:
                sample = prev.lerp(point, factor)
                if point_inside_collider(sample):
                    return "sample", sample, nearest_normal(sample)
            return None

        def repair_final_issue(point: Vector, issue) -> Vector:
            kind, loc, normal = issue
            move = point - prev
            direction = move.normalized() if move.length > 1.0e-9 else fallback_dir
            push_dir = push_direction(loc, normal)
            if kind == "ray":
                slide = _project_to_tangent(direction, push_dir)
                remaining = max(0.0, seg_len - (loc - prev).length)
                return project_to_length(loc + push_dir * collision_radius_m + slide * remaining)
            repaired = pushed_out_point(loc, normal, collision_radius_m)
            tangent = _project_to_tangent(direction, push_dir)
            return project_to_length(repaired + tangent * 0.0005)

        def find_safe_candidate(point: Vector) -> Vector:
            raw_dirs = [
                point - prev,
                DOWN.copy(),
                back_down.copy(),
                BACK.copy(),
                (BACK * 0.30 + DOWN * 0.95),
            ]
            for probe in (point, prev.lerp(point, 0.5)):
                normal = nearest_normal(probe)
                if normal is None:
                    continue
                push_dir = push_direction(probe, normal)
                raw_dirs.extend((
                    push_dir,
                    _project_to_tangent(DOWN, push_dir),
                    _project_to_tangent(BACK, push_dir),
                    DOWN * 0.78 + push_dir * 0.35,
                ))

            tried = []
            for raw in raw_dirs:
                if raw.length <= 1.0e-9:
                    continue
                direction = raw.normalized()
                if any((direction - existing).length < 1.0e-4 for existing in tried):
                    continue
                tried.append(direction)
                candidate = prev + direction * seg_len
                if final_segment_issue(candidate) is None:
                    return candidate
            return point

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
                        push_dir = push_direction(loc, normal)
                        slide = _project_to_tangent(direction, push_dir)
                        candidate = loc + push_dir * collision_radius_m + slide * max(0.0, seg_len - dist)
                        v = candidate - prev
                        candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                        stats["ray_hits"] += 1

            if point_inside_collider(candidate):
                candidate = pushed_out_point(candidate, None, collision_radius_m)
                v = candidate - prev
                candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                stats["inside_pushes"] += 1

            nearest = bvh.find_nearest(candidate, follow_radius_m)
            if nearest is not None:
                loc, normal, _index, dist = nearest
                if loc is not None and normal is not None and dist is not None:
                    stats["min_clearance_m"] = min(stats["min_clearance_m"], float(dist))
                    inside = point_inside_collider(candidate)
                    if inside or dist < collision_radius_m:
                        push_dir = push_direction(candidate if inside else loc, normal)
                        tangent = _project_to_tangent(candidate - prev, push_dir)
                        if inside:
                            candidate = pushed_out_point(candidate, normal, collision_radius_m) + tangent * 0.0005
                        else:
                            push_distance = collision_radius_m - dist
                            candidate = candidate + push_dir * max(push_distance, collision_radius_m * 0.5)
                            candidate = candidate + tangent * 0.0005
                        v = candidate - prev
                        candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                        if inside:
                            stats["inside_pushes"] += 1
                        else:
                            stats["nearest_pushes"] += 1

            mid = prev.lerp(candidate, 0.5)
            if point_inside_collider(mid):
                candidate = pushed_out_point(mid, None, collision_radius_m)
                v = candidate - prev
                candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                stats["inside_pushes"] += 1
                mid = prev.lerp(candidate, 0.5)

            nearest_mid = bvh.find_nearest(mid, follow_radius_m)
            if nearest_mid is not None:
                loc, normal, _index, dist = nearest_mid
                if normal is not None and dist is not None:
                    stats["min_clearance_m"] = min(stats["min_clearance_m"], float(dist))
                    inside = point_inside_collider(mid)
                    if inside or dist < collision_radius_m:
                        push_dir = push_direction(mid if inside else loc, normal)
                        if inside:
                            candidate = pushed_out_point(mid, normal, collision_radius_m)
                        else:
                            push_distance = collision_radius_m - dist
                            candidate = candidate + push_dir * max(push_distance, collision_radius_m * 0.5)
                        v = candidate - prev
                        candidate = prev + (v.normalized() if v.length > 1.0e-9 else fallback_dir) * seg_len
                        if inside:
                            stats["inside_pushes"] += 1
                        else:
                            stats["sample_pushes"] += 1

        candidate = project_to_length(candidate)
        for _guard in range(4):
            issue = final_segment_issue(candidate)
            if issue is None:
                return candidate
            candidate = repair_final_issue(candidate, issue)
            stats["final_guard_pushes"] += 1

        safe = find_safe_candidate(candidate)
        if final_segment_issue(safe) is None:
            stats["final_guard_fallbacks"] += 1
            return safe
        stats["final_guard_unresolved"] += 1
        return safe

    for si in target_strands:
        start, count = spans[si]
        old = [world_pts[start + j].copy() for j in range(count)]
        seg_lens = []
        for j in range(count - 1):
            length = (old[j + 1] - old[j]).length
            seg_lens.append(length if length > 1.0e-6 else 0.01)

        def root_emergence_direction(j: int, base_dir: Vector):
            if j >= 1 or j + 1 >= len(old):
                return base_dir, False
            original = old[j + 1] - old[j]
            if original.length <= 1.0e-9:
                return base_dir, False
            original.normalize()

            is_head_region = old[j].z >= head_region_min_z
            is_outward = is_head_region and original.z > 0.20
            nearest = bvh.find_nearest(old[j], follow_radius_m)
            if nearest is not None:
                _loc, normal, _index, _dist = nearest
                if normal is not None and normal.length > 1.0e-7:
                    normal = outward_head_normal(old[j], normal, root_outward_dirs[si])
                    is_outward = is_outward or original.dot(normal) > 0.10
            if not is_outward:
                return base_dir, False

            weight = 1.0 if j == 0 else 0.65
            mixed = original * weight + base_dir * (1.0 - weight)
            if mixed.length <= 1.0e-7:
                mixed = original
            else:
                mixed.normalize()
            stats["root_emergence_guides"] += 1
            return mixed, True

        def first_rod_normal_lock():
            if not seg_lens or old[0].z < head_region_min_z:
                return None
            nearest = bvh.find_nearest(old[0], follow_radius_m)
            if nearest is None:
                return None
            loc, normal, _index, _dist = nearest
            if loc is None or normal is None or normal.length <= 1.0e-7:
                return None
            normal = outward_head_normal(old[0], normal, root_outward_dirs[si])
            stats["normal_root_locks"] += 1
            return old[0] + normal * seg_lens[0]

        new = [old[0].copy()]
        locked_first_rod = False
        first_tip = first_rod_normal_lock()
        if first_tip is not None:
            new.append(first_tip)
            locked_first_rod = True

        def lower_free_direction(prev_dir: Vector | None) -> Vector:
            direction = DOWN.copy()
            if prev_dir is not None and prev_dir.length > 1.0e-9:
                follow = prev_dir.normalized()
                if follow.z > 0.0:
                    follow.z = 0.0
                    if follow.length <= 1.0e-9:
                        follow = DOWN.copy()
                    else:
                        follow.normalize()
                direction = DOWN * 0.82 + follow * 0.18
            if direction.length <= 1.0e-9:
                direction = DOWN.copy()
            direction.normalize()
            return direction

        def segment_in_lower_free_region(j: int, anchor: Vector, end: Vector | None = None) -> bool:
            if point_in_lower_free_region(anchor):
                return True
            if end is not None and point_in_lower_free_region(end):
                return True
            if j + 1 < len(old) and point_in_lower_free_region(old[j + 1]):
                return True
            return False

        surface_run = 0.0
        start_joint = 1 if locked_first_rod else 0
        for j in range(start_joint, len(seg_lens)):
            seg_len = seg_lens[j]
            t = j / max(1, len(seg_lens) - 1)
            lower_free = segment_in_lower_free_region(j, new[-1])
            prev_dir = new[-1] - new[-2] if len(new) > 1 else None
            if lower_free:
                desired_dir = lower_free_direction(prev_dir)
                has_emergence = False
                stats["lower_free_rods"] += 1
            else:
                desired_dir, has_emergence = root_emergence_direction(j, _base_drop_direction(t))
            direction, surface_run, in_surface = choose_direction(new[-1], desired_dir, surface_run, lower_free=lower_free)
            if not in_surface:
                surface_run = 0.0
            if not lower_free:
                direction, limited = _limit_turn_direction(prev_dir, direction, max_turn_angle_rad)
                if limited:
                    stats["angle_limited_rods"] += 1
            new.append(solve_candidate(new[-1], new[-1] + direction * seg_len, seg_len, shallow_guard=has_emergence))

        for _pass in range(3):
            surface_run = 0.0
            for j, seg_len in enumerate(seg_lens):
                if locked_first_rod and j == 0:
                    continue
                base_dir = _base_drop_direction(j / max(1, len(seg_lens) - 1))
                lower_free = segment_in_lower_free_region(j, new[j], new[j + 1])
                prev_dir = new[j] - new[j - 1] if j > 0 else None
                if lower_free:
                    has_emergence = False
                    direction = lower_free_direction(prev_dir)
                    stats["lower_free_rods"] += 1
                else:
                    emergence_dir, has_emergence = root_emergence_direction(j, base_dir)
                    direction = new[j + 1] - new[j]
                    if direction.length <= 1.0e-9:
                        direction = emergence_dir
                    else:
                        direction.normalize()
                        if has_emergence:
                            direction = emergence_dir
                        elif direction.z > -0.35:
                            direction = base_dir
                direction, surface_run, in_surface = choose_direction(new[j], direction, surface_run, lower_free=lower_free)
                if not in_surface:
                    surface_run = 0.0
                if not lower_free:
                    direction, limited = _limit_turn_direction(prev_dir, direction, max_turn_angle_rad)
                    if limited:
                        stats["angle_limited_rods"] += 1
                new[j + 1] = solve_candidate(new[j], new[j] + direction * seg_len, seg_len, shallow_guard=has_emergence)

        if len(new) > 1:
            tip_dir = new[-1] - new[-2]
            if tip_dir.length > 1.0e-9:
                stats["tip_down_dot_sum"] += tip_dir.normalized().dot(DOWN)

        for j in range(count):
            idx = start + j
            stats["max_move_m"] = max(stats["max_move_m"], (new[j] - old[j]).length)
            world_pts[idx] = new[j]
            stats["changed_points"] += 1
            if point_inside_collider(new[j]):
                stats["remaining_close_points"] += 1
                continue
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
        "back_release_events": int(stats["back_release_events"]),
        "release_blocked_inside": int(stats["release_blocked_inside"]),
        "release_blocked_ray": int(stats["release_blocked_ray"]),
        "forced_release_events": int(stats["forced_release_events"]),
        "ray_hits": int(stats["ray_hits"]),
        "nearest_pushes": int(stats["nearest_pushes"]),
        "inside_pushes": int(stats["inside_pushes"]),
        "head_radial_pushes": int(stats["head_radial_pushes"]),
        "sample_pushes": int(stats["sample_pushes"]),
        "final_guard_pushes": int(stats["final_guard_pushes"]),
        "final_guard_fallbacks": int(stats["final_guard_fallbacks"]),
        "final_guard_unresolved": int(stats["final_guard_unresolved"]),
        "root_emergence_guides": int(stats["root_emergence_guides"]),
        "normal_root_locks": int(stats["normal_root_locks"]),
        "angle_limited_rods": int(stats["angle_limited_rods"]),
        "lower_free_rods": int(stats["lower_free_rods"]),
        "remaining_close_points": int(stats["remaining_close_points"]),
        "min_clearance_mm": None if stats["min_clearance_m"] == 999.0 else stats["min_clearance_m"] * 1000.0,
        "max_move_cm": stats["max_move_m"] * 100.0,
        "max_length_error_mm": stats["max_len_error_m"] * 1000.0,
        "avg_tip_down_dot": stats["tip_down_dot_sum"] / max(1, processed),
    }
