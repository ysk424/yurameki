"""NVIDIA Warp joint-chain simulation for Yurameki.

This path intentionally uses the existing Blender Curves joints. It does not
resample into cylinders. The first N points of every strand are kinematic
constraints following the evaluated Curves shape; the remaining points are
simulated with distance, bend, gravity, damping, and Warp Mesh collision.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass

from bpy.app.handlers import persistent
import bpy
from mathutils import Vector
import numpy as np
import warp as wp


@dataclass
class WarpCheckStats:
    n_strands: int
    points_per_strand: int
    n_points: int
    n_segments: int
    root_locked_points: int
    n_vertices: int
    n_triangles: int
    device: str
    device_name: str
    device_arch: int
    warp_version: str


@dataclass
class WarpSimStats:
    start_frame: int
    end_frame: int
    n_frames: int
    n_strands: int
    simulated_strands: int
    guide_decimation: int
    points_per_strand: int
    root_locked_points: int
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
    post_keep_collision_hits: int
    post_keep_active_strands: int
    body_hard_guard_points: int
    body_fk_repair_strands: int
    body_fk_repair_points: int
    body_fk_escape_points: int
    body_fk_failed_points: int
    body_fk_velocity_zeroed: int
    keep_length: bool
    keep_length_source_frame: int
    max_keep_length_error_mm: float
    cache_path: str
    device: str
    device_name: str
    device_arch: int
    elapsed_sec: float


@dataclass
class YuramekiRuntimeCache:
    object_name: str
    data_name: str
    frames: np.ndarray
    local_values: np.ndarray
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


@dataclass
class BodyFkRepairStats:
    strands: int = 0
    points: int = 0
    escape_points: int = 0
    failed_points: int = 0


_CACHE_REGISTRY: dict[str, YuramekiRuntimeCache] = {}
_CACHE_MUTED = False
POST_KEEP_CONTACT_TTL_FRAMES = 3
BODY_FK_REPAIR_TTL_FRAMES = 3
BODY_FK_SEARCH_T_VALUES = (0.25, 0.50, 0.75, 1.00)
BODY_FK_BINARY_STEPS = 3


@wp.func
def _normalize_or(v: wp.vec3, fallback: wp.vec3):
    length = wp.length(v)
    if length > 1.0e-9:
        return v / length
    return fallback


@wp.kernel
def _predict_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    target_start: wp.array(dtype=wp.vec3),
    target_end: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    alpha: float,
    dt: float,
    gravity_x: float,
    gravity_y: float,
    gravity_z: float,
    max_velocity: float,
):
    i = wp.tid()
    if inv_mass[i] <= 0.0:
        target = target_start[i] * (1.0 - alpha) + target_end[i] * alpha
        pos[i] = target
        predicted[i] = target
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v = vel[i] + wp.vec3(gravity_x, gravity_y, gravity_z) * dt
        if max_velocity > 0.0:
            speed = wp.length(v)
            if speed > max_velocity and speed > 1.0e-9:
                v = v / speed * max_velocity
        vel[i] = v
        predicted[i] = pos[i] + v * dt


@wp.kernel
def _solve_distance_kernel(
    predicted: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest: wp.array(dtype=float),
    points_per_strand: int,
    parity: int,
    compliance: float,
    dt: float,
):
    segment_id = wp.tid()
    segments_per_strand = points_per_strand - 1
    local = segment_id % segments_per_strand
    if local % 2 != parity:
        return

    strand = segment_id // segments_per_strand
    i = strand * points_per_strand + local
    j = i + 1
    wi = inv_mass[i]
    wj = inv_mass[j]
    wsum = wi + wj
    if wsum <= 1.0e-12:
        return

    delta = predicted[i] - predicted[j]
    length = wp.length(delta)
    if length <= 1.0e-8:
        return

    constraint = length - rest[segment_id]
    alpha = compliance / (dt * dt)
    dlambda = -constraint / (wsum + alpha)
    grad = delta / length
    predicted[i] = predicted[i] + wi * dlambda * grad
    predicted[j] = predicted[j] - wj * dlambda * grad


@wp.kernel
def _solve_bend_kernel(
    predicted: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest: wp.array(dtype=float),
    points_per_strand: int,
    parity: int,
    compliance: float,
    dt: float,
):
    if points_per_strand < 3:
        return
    bend_id = wp.tid()
    bends_per_strand = points_per_strand - 2
    local = bend_id % bends_per_strand
    if local % 2 != parity:
        return

    strand = bend_id // bends_per_strand
    i = strand * points_per_strand + local
    j = i + 2
    wi = inv_mass[i]
    wj = inv_mass[j]
    wsum = wi + wj
    if wsum <= 1.0e-12:
        return

    delta = predicted[i] - predicted[j]
    length = wp.length(delta)
    if length <= 1.0e-8:
        return

    constraint = length - rest[bend_id]
    alpha = compliance / (dt * dt)
    dlambda = -constraint / (wsum + alpha)
    grad = delta / length
    predicted[i] = predicted[i] + wi * dlambda * grad
    predicted[j] = predicted[j] - wj * dlambda * grad


@wp.kernel
def _derive_velocity_kernel(
    pos: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    contact_mask: wp.array(dtype=wp.int32),
    dt: float,
    damping: float,
    max_velocity: float,
    collision_velocity_damping: float,
):
    i = wp.tid()
    if inv_mass[i] <= 0.0:
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
    else:
        v = (predicted[i] - pos[i]) / dt * (1.0 - damping)
        if max_velocity > 0.0:
            speed = wp.length(v)
            if speed > max_velocity and speed > 1.0e-9:
                v = v / speed * max_velocity
        if contact_mask[i] != 0:
            contact_keep = 1.0 - collision_velocity_damping
            if contact_keep < 0.0:
                contact_keep = 0.0
            v = v * contact_keep
        vel[i] = v


@wp.kernel
def _clear_contact_mask_kernel(contact_mask: wp.array(dtype=wp.int32)):
    i = wp.tid()
    contact_mask[i] = 0


@wp.kernel
def _commit_kernel(
    pos: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pos[i] = predicted[i]


@dataclass
class ColliderMeshSet:
    body: wp.Mesh | None
    clothes: wp.Mesh | None
    n_vertices: int
    n_triangles: int


@wp.kernel
def _body_point_collision_kernel(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    points_per_strand: int,
    margin: float,
    search_distance: float,
    max_correction: float,
    collision_response: float,
    allow_sweep: int,
    contact_mask: wp.array(dtype=wp.int32),
    frame_contact_strands: wp.array(dtype=wp.int32),
    hit_count: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if inv_mass[i] <= 0.0:
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    old_position = positions[i]
    new_position = predicted[i]
    delta = new_position - old_position
    distance = wp.length(delta)
    normal = wp.vec3(0.0, 0.0, 0.0)
    contacted = 0

    if allow_sweep == 1 and distance > 1.0e-9:
        direction = delta / distance
        ray = wp.mesh_query_ray(mesh, old_position, direction, distance)
        if ray.result and wp.dot(delta, ray.normal) < 0.0:
            normal = ray.normal
            target = old_position + direction * ray.t + normal * margin
            correction = target - new_position
            correction_length = wp.length(correction)
            if correction_length > max_correction and correction_length > 1.0e-9:
                correction = correction / correction_length * max_correction
            new_position = new_position + correction * collision_response
            contacted = 1

    if contacted == 0:
        query = wp.mesh_query_point_sign_normal(mesh, new_position, search_distance, 1.0e-3)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            surface_delta = new_position - closest
            unsigned_distance = wp.length(surface_delta)
            signed_distance = unsigned_distance * query.sign
            error = signed_distance - margin
            if error < 0.0:
                if unsigned_distance > 1.0e-9:
                    normal = wp.normalize(surface_delta) * query.sign
                else:
                    normal = wp.mesh_eval_face_normal(mesh, query.face)
                correction = -normal * error
                correction_length = wp.length(correction)
                if correction_length > max_correction and correction_length > 1.0e-9:
                    correction = correction / correction_length * max_correction
                new_position = new_position + correction * collision_response
                contacted = 1

    if contacted == 1:
        contact_mask[i] = 1
        frame_contact_strands[i // points_per_strand] = 1
        velocity = velocities[i]
        normal_speed = wp.dot(velocity, normal)
        if normal_speed < 0.0:
            velocities[i] = velocity - normal * normal_speed
        predicted[i] = new_position
        wp.atomic_add(hit_count, 0, 1)


@wp.kernel
def _cloth_point_collision_kernel(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    points_per_strand: int,
    margin: float,
    search_distance: float,
    max_correction: float,
    collision_response: float,
    allow_sweep: int,
    contact_mask: wp.array(dtype=wp.int32),
    frame_contact_strands: wp.array(dtype=wp.int32),
    hit_count: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if inv_mass[i] <= 0.0:
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    old_position = positions[i]
    new_position = predicted[i]
    delta = new_position - old_position
    distance = wp.length(delta)
    normal = wp.vec3(0.0, 0.0, 0.0)
    contacted = 0

    if allow_sweep == 1 and distance > 1.0e-9:
        direction = delta / distance
        ray = wp.mesh_query_ray(mesh, old_position, direction, distance)
        if ray.result:
            normal = ray.normal
            if wp.dot(delta, normal) > 0.0:
                normal = -normal
            target = old_position + direction * ray.t + normal * margin
            correction = target - new_position
            correction_length = wp.length(correction)
            if correction_length > max_correction and correction_length > 1.0e-9:
                correction = correction / correction_length * max_correction
            new_position = new_position + correction * collision_response
            contacted = 1

    if contacted == 0:
        query = wp.mesh_query_point_no_sign(mesh, new_position, search_distance)
        if query.result:
            closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            surface_delta = new_position - closest
            unsigned_distance = wp.length(surface_delta)
            if unsigned_distance < margin:
                normal = wp.mesh_eval_face_normal(mesh, query.face)
                if unsigned_distance > 1.0e-9 and wp.dot(surface_delta, normal) < 0.0:
                    normal = -normal
                target = closest + normal * margin
                correction = target - new_position
                correction_length = wp.length(correction)
                if correction_length > max_correction and correction_length > 1.0e-9:
                    correction = correction / correction_length * max_correction
                new_position = new_position + correction * collision_response
                contacted = 1

    if contacted == 1:
        contact_mask[i] = 1
        frame_contact_strands[i // points_per_strand] = 1
        velocity = velocities[i]
        normal_speed = wp.dot(velocity, normal)
        if normal_speed < 0.0:
            velocities[i] = velocity - normal * normal_speed
        predicted[i] = new_position
        wp.atomic_add(hit_count, 0, 1)


@wp.kernel
def _segment_collision_kernel(
    mesh: wp.uint64,
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    points_per_strand: int,
    margin: float,
    max_correction: float,
    collision_response: float,
    parity: int,
    contact_mask: wp.array(dtype=wp.int32),
    frame_contact_strands: wp.array(dtype=wp.int32),
    hit_count: wp.array(dtype=wp.int32),
):
    segment_id = wp.tid()
    segments_per_strand = points_per_strand - 1
    local = segment_id % segments_per_strand
    if local % 2 != parity:
        return

    strand = segment_id // segments_per_strand
    i = strand * points_per_strand + local
    j = i + 1
    p0 = predicted[i]
    p1 = predicted[j]
    delta = p1 - p0
    distance = wp.length(delta)
    if distance <= 1.0e-9:
        return

    direction = delta / distance
    ray = wp.mesh_query_ray(mesh, p0, direction, distance)
    if not ray.result or ray.t <= 1.0e-6 or ray.t >= distance - 1.0e-6:
        return

    normal = ray.normal
    if wp.dot(delta, normal) > 0.0:
        normal = -normal
    target = p0 + direction * ray.t + normal * margin
    correction = target - p1
    correction_length = wp.length(correction)
    if correction_length > max_correction and correction_length > 1.0e-9:
        correction = correction / correction_length * max_correction

    if inv_mass[j] > 0.0:
        predicted[j] = p1 + correction * collision_response
        contact_mask[j] = 1
        frame_contact_strands[strand] = 1
        velocity = velocities[j]
        normal_speed = wp.dot(velocity, normal)
        if normal_speed < 0.0:
            velocities[j] = velocity - normal * normal_speed
        wp.atomic_add(hit_count, 0, 1)


@wp.kernel
def _post_keep_body_collision_kernel(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    active_strands: wp.array(dtype=wp.int32),
    rest: wp.array(dtype=float),
    points_per_strand: int,
    margin: float,
    search_distance: float,
    max_correction: float,
    collision_response: float,
    contact_strands: wp.array(dtype=wp.int32),
    hit_count: wp.array(dtype=wp.int32),
):
    strand = wp.tid()
    if active_strands[strand] == 0:
        return

    segments_per_strand = int(points_per_strand - 1)
    local = int(0)
    while local < segments_per_strand:
        i = strand * points_per_strand + local
        j = i + 1
        if inv_mass[j] > 0.0:
            p0 = positions[i]
            p1 = positions[j]
            rest_length = rest[strand * segments_per_strand + local]
            if rest_length > 1.0e-8:
                direction = _normalize_or(p1 - p0, wp.vec3(0.0, 0.0, -1.0))
                p1 = p0 + direction * rest_length
                contacted = int(0)

                ray = wp.mesh_query_ray(mesh, p0, direction, rest_length)
                if ray.result and ray.t > 1.0e-6 and ray.t < rest_length - 1.0e-6:
                    normal = ray.normal
                    if wp.dot(direction, normal) > 0.0:
                        normal = -normal
                    target = p0 + direction * ray.t + normal * margin
                    correction = target - p1
                    correction_length = wp.length(correction)
                    if correction_length > max_correction and correction_length > 1.0e-9:
                        correction = correction / correction_length * max_correction
                    candidate = p1 + correction * collision_response
                    direction = _normalize_or(candidate - p0, direction)
                    p1 = p0 + direction * rest_length
                    contacted = int(1)

                query = wp.mesh_query_point_sign_normal(mesh, p1, search_distance, 1.0e-3)
                if query.result:
                    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                    surface_delta = p1 - closest
                    unsigned_distance = wp.length(surface_delta)
                    signed_distance = unsigned_distance * query.sign
                    error = signed_distance - margin
                    if error < 0.0:
                        if unsigned_distance > 1.0e-9:
                            normal = wp.normalize(surface_delta) * query.sign
                        else:
                            normal = wp.mesh_eval_face_normal(mesh, query.face)
                        correction = -normal * error
                        correction_length = wp.length(correction)
                        if correction_length > max_correction and correction_length > 1.0e-9:
                            correction = correction / correction_length * max_correction
                        candidate = p1 + correction * collision_response
                        direction = _normalize_or(candidate - p0, direction)
                        p1 = p0 + direction * rest_length
                        contacted = int(1)

                positions[j] = p1
                if contacted == 1:
                    contact_strands[strand] = 1
                    wp.atomic_add(hit_count, 0, 1)
        local += 1


@wp.kernel
def _post_keep_cloth_collision_kernel(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    active_strands: wp.array(dtype=wp.int32),
    rest: wp.array(dtype=float),
    points_per_strand: int,
    margin: float,
    search_distance: float,
    max_correction: float,
    collision_response: float,
    contact_strands: wp.array(dtype=wp.int32),
    hit_count: wp.array(dtype=wp.int32),
):
    strand = wp.tid()
    if active_strands[strand] == 0:
        return

    segments_per_strand = int(points_per_strand - 1)
    local = int(0)
    while local < segments_per_strand:
        i = strand * points_per_strand + local
        j = i + 1
        if inv_mass[j] > 0.0:
            p0 = positions[i]
            p1 = positions[j]
            rest_length = rest[strand * segments_per_strand + local]
            if rest_length > 1.0e-8:
                direction = _normalize_or(p1 - p0, wp.vec3(0.0, 0.0, -1.0))
                p1 = p0 + direction * rest_length
                contacted = int(0)

                ray = wp.mesh_query_ray(mesh, p0, direction, rest_length)
                if ray.result and ray.t > 1.0e-6 and ray.t < rest_length - 1.0e-6:
                    normal = ray.normal
                    if wp.dot(direction, normal) > 0.0:
                        normal = -normal
                    target = p0 + direction * ray.t + normal * margin
                    correction = target - p1
                    correction_length = wp.length(correction)
                    if correction_length > max_correction and correction_length > 1.0e-9:
                        correction = correction / correction_length * max_correction
                    candidate = p1 + correction * collision_response
                    direction = _normalize_or(candidate - p0, direction)
                    p1 = p0 + direction * rest_length
                    contacted = int(1)

                query = wp.mesh_query_point_no_sign(mesh, p1, search_distance)
                if query.result:
                    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                    surface_delta = p1 - closest
                    unsigned_distance = wp.length(surface_delta)
                    if unsigned_distance < margin:
                        normal = wp.mesh_eval_face_normal(mesh, query.face)
                        if unsigned_distance > 1.0e-9 and wp.dot(surface_delta, normal) < 0.0:
                            normal = -normal
                        target = closest + normal * margin
                        correction = target - p1
                        correction_length = wp.length(correction)
                        if correction_length > max_correction and correction_length > 1.0e-9:
                            correction = correction / correction_length * max_correction
                        candidate = p1 + correction * collision_response
                        direction = _normalize_or(candidate - p0, direction)
                        p1 = p0 + direction * rest_length
                        contacted = int(1)

                positions[j] = p1
                if contacted == 1:
                    contact_strands[strand] = 1
                    wp.atomic_add(hit_count, 0, 1)
        local += 1


@wp.kernel
def _zero_contact_strand_velocity_kernel(
    velocities: wp.array(dtype=wp.vec3),
    contact_strands: wp.array(dtype=wp.int32),
    points_per_strand: int,
):
    i = wp.tid()
    if contact_strands[i // points_per_strand] != 0:
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)


def _curve_spans(curves_obj):
    return [
        (int(curve.first_point_index), int(curve.points_length))
        for curve in curves_obj.data.curves
    ]


def _read_world_points(data_owner, n_total: int, matrix_world):
    attr = data_owner.attributes.get("position")
    if attr is None or len(attr.data) != n_total:
        return None
    flat = np.zeros(n_total * 3, dtype=np.float32)
    attr.data.foreach_get("vector", flat)
    local_pts = flat.reshape(n_total, 3)
    mw = np.array(matrix_world, dtype=np.float32)
    homogeneous = np.column_stack([local_pts, np.ones(n_total, dtype=np.float32)])
    return (homogeneous @ mw.T)[:, :3].astype(np.float32, copy=True)


def _read_world(curves_obj):
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    n_total = len(attr.data)
    if n_total == 0:
        raise ValueError("Curves has no points")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = curves_obj.evaluated_get(depsgraph)
    world = _read_world_points(evaluated.data, n_total, evaluated.matrix_world)
    original = _read_world_points(curves_obj.data, n_total, curves_obj.matrix_world)
    if world is None or original is None:
        raise ValueError("Could not read world-space Curves positions")
    return world.astype(np.float32, copy=False), original.astype(np.float32, copy=False)


def _write_world_points(curves_obj, world_pts: np.ndarray, offset=None) -> None:
    local_pts = _world_to_local_points(curves_obj, world_pts, offset=offset)
    attr = curves_obj.data.attributes.get("position")
    if attr is None or len(attr.data) != len(world_pts):
        raise ValueError("Curves position attribute shape changed")
    attr.data.foreach_set("vector", local_pts.ravel())
    curves_obj.data.update_tag()


def _force_viewport_refresh() -> None:
    bpy.context.view_layer.update()
    if bpy.app.background:
        return
    try:
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:
        pass


def _show_sim_frame(curves_obj, frame: int, world_pts: np.ndarray, offset=None,
                    pps: int | None = None,
                    keep_rest_lengths: np.ndarray | None = None,
                    keep_fallback_dirs: np.ndarray | None = None,
                    body_hard_guard_obj=None,
                    body_hard_guard_margin: float = 0.0,
                    body_hard_guard_seed=None,
                    body_fk_root_locked_points: int = 1,
                    correction_passes: int = 3) -> float:
    bpy.context.scene.frame_set(int(frame))
    _write_world_points(curves_obj, world_pts, offset=offset)
    max_error = 0.0
    if pps is not None and keep_rest_lengths is not None and keep_fallback_dirs is not None:
        for _ in range(max(1, int(correction_passes))):
            bpy.context.view_layer.update()
            eval_world, original_world = _read_world(curves_obj)
            corrected, keep_error = _keep_length_fk(
                eval_world,
                int(pps),
                keep_rest_lengths,
                keep_fallback_dirs,
            )
            max_error = max(max_error, keep_error)
            _write_world_points(curves_obj, corrected, offset=eval_world - original_world)
    if body_hard_guard_obj is not None and body_hard_guard_seed is not None:
        bpy.context.view_layer.update()
        eval_world, original_world = _read_world(curves_obj)
        if pps is not None:
            guarded, fixed, guard_strands = _apply_body_seed_hard_guard(
                eval_world,
                body_hard_guard_obj,
                body_hard_guard_margin,
                body_hard_guard_seed,
                points_per_strand=int(pps),
            )
            if fixed and keep_rest_lengths is not None and keep_fallback_dirs is not None:
                guarded, _repair_stats, _repair_strands = _apply_body_fk_hard_repair(
                    guarded,
                    int(pps),
                    keep_rest_lengths,
                    keep_fallback_dirs,
                    body_hard_guard_obj,
                    body_hard_guard_margin,
                    body_hard_guard_seed,
                    root_locked_points=int(body_fk_root_locked_points),
                    active_strands=guard_strands,
                )
        else:
            guarded, _fixed = _apply_body_seed_hard_guard(
                eval_world,
                body_hard_guard_obj,
                body_hard_guard_margin,
                body_hard_guard_seed,
            )
        _write_world_points(curves_obj, guarded, offset=eval_world - original_world)
    _force_viewport_refresh()
    return max_error


def _world_to_local_points(curves_obj, world_pts: np.ndarray, offset=None) -> np.ndarray:
    n_total = len(world_pts)
    write_pts = world_pts if offset is None else world_pts - offset
    mw_inv = np.array(curves_obj.matrix_world.inverted(), dtype=np.float32)
    homogeneous = np.column_stack([write_pts, np.ones(n_total, dtype=np.float32)])
    return (homogeneous @ mw_inv.T)[:, :3].astype(np.float32, copy=True)


def _uniform_points_per_strand(curves_obj) -> tuple[int, int]:
    spans = _curve_spans(curves_obj)
    if not spans:
        raise ValueError("Curves object has no strands")
    lengths = sorted({count for _start, count in spans})
    if len(lengths) != 1:
        raise ValueError(f"Warp path requires uniform points per strand, got {lengths[:8]}")
    if lengths[0] < 3:
        raise ValueError("Warp path needs at least 3 points per strand")
    return int(lengths[0]), len(spans)


def _evaluated_mesh_arrays(objects) -> tuple[np.ndarray, np.ndarray]:
    vertices_out = []
    indices_out = []
    offset = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj is None or obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            vertex_count = len(mesh.vertices)
            triangle_count = len(mesh.loop_triangles)
            if vertex_count == 0 or triangle_count == 0:
                continue
            verts = np.empty(vertex_count * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", verts)
            verts = verts.reshape(-1, 3)
            matrix = np.array(evaluated.matrix_world, dtype=np.float32)
            homogeneous = np.column_stack((verts, np.ones(vertex_count, dtype=np.float32)))
            verts = (homogeneous @ matrix.T)[:, :3].astype(np.float32, copy=False)
            tris = np.empty(triangle_count * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tris)
            tris = tris.reshape(-1, 3) + offset
            vertices_out.append(verts)
            indices_out.append(tris)
            offset += vertex_count
        finally:
            evaluated.to_mesh_clear()
    if not vertices_out or not indices_out:
        raise ValueError("Collider meshes have no evaluated triangles")
    vertices = np.ascontiguousarray(np.vstack(vertices_out), dtype=np.float32)
    indices = np.ascontiguousarray(np.vstack(indices_out).reshape(-1), dtype=np.int32)
    return vertices, indices


def _armature_from_object(obj):
    if obj is None:
        return None
    parent = getattr(obj, "parent", None)
    if parent is not None and parent.type == "ARMATURE":
        return parent
    for mod in getattr(obj, "modifiers", ()):
        if mod.type == "ARMATURE" and getattr(mod, "object", None) is not None:
            return mod.object
    return None


def _head_seed_world(body_obj):
    armature = _armature_from_object(body_obj)
    if armature is None:
        return None
    candidates = (
        "CC_Base_Head",
        "Head",
        "head",
    )
    bone_name = None
    for name in candidates:
        if name in armature.pose.bones or name in armature.data.bones:
            bone_name = name
            break
    if bone_name is None:
        for bone in armature.data.bones:
            if "head" in bone.name.lower():
                bone_name = bone.name
                break
    if bone_name is None:
        return None
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is not None:
        return armature.matrix_world @ pose_bone.tail
    bone = armature.data.bones.get(bone_name)
    return armature.matrix_world @ bone.tail_local if bone is not None else None


@dataclass
class _BodyRayContext:
    evaluated: object
    matrix: object
    matrix_inv: object
    normal_matrix: object
    margin: float
    body_span: float


def _body_ray_context(body_obj, margin: float):
    if body_obj is None or body_obj.type != "MESH":
        return None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = body_obj.evaluated_get(depsgraph)
    matrix = evaluated.matrix_world.copy()
    dims = getattr(body_obj, "dimensions", (1.0, 1.0, 1.0))
    return _BodyRayContext(
        evaluated=evaluated,
        matrix=matrix,
        matrix_inv=matrix.inverted(),
        normal_matrix=matrix.to_3x3(),
        margin=max(float(margin), 0.0),
        body_span=max(float(max(dims)), 0.5),
    )


def _vector_np(v: Vector) -> np.ndarray:
    return np.asarray((float(v.x), float(v.y), float(v.z)), dtype=np.float32)


def _normalize_np(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length > 1.0e-8:
        return (v / length).astype(np.float32, copy=False)
    fallback_length = float(np.linalg.norm(fallback))
    if fallback_length > 1.0e-8:
        return (fallback / fallback_length).astype(np.float32, copy=False)
    return np.asarray((0.0, 0.0, -1.0), dtype=np.float32)


def _slerp_direction_np(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = _normalize_np(np.asarray(a, dtype=np.float32), np.asarray((0.0, 0.0, -1.0), dtype=np.float32))
    b = _normalize_np(np.asarray(b, dtype=np.float32), a)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_np((1.0 - float(t)) * a + float(t) * b, b)
    if dot < -0.9995:
        return b.copy()
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - float(t)) * theta) / sin_theta
    wb = math.sin(float(t) * theta) / sin_theta
    return _normalize_np(wa * a + wb * b, b)


def _append_unique_seed(seeds: list[np.ndarray], seed_world) -> None:
    if seed_world is None:
        return
    seed = _vector_np(Vector(seed_world))
    for existing in seeds:
        if float(np.linalg.norm(existing - seed)) < 1.0e-5:
            return
    seeds.append(seed)


def _body_internal_seed_points(body_obj, preferred_seed=None) -> np.ndarray:
    seeds: list[np.ndarray] = []
    _append_unique_seed(seeds, preferred_seed)
    armature = _armature_from_object(body_obj)
    if armature is not None:
        terms = ("head", "neck", "spine", "chest")
        pose_bones = getattr(armature.pose, "bones", ())
        for pose_bone in pose_bones:
            lower = pose_bone.name.lower()
            if not any(term in lower for term in terms):
                continue
            _append_unique_seed(seeds, armature.matrix_world @ pose_bone.head)
            _append_unique_seed(seeds, armature.matrix_world @ pose_bone.tail)
    if not seeds:
        _append_unique_seed(seeds, _head_seed_world(body_obj))
    if not seeds:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(seeds), dtype=np.float32)


def _choose_body_seed(point: np.ndarray,
                      strand_root: np.ndarray | None,
                      seeds: np.ndarray) -> Vector | None:
    if seeds is None or len(seeds) == 0:
        return None
    ref = np.asarray(strand_root if strand_root is not None else point, dtype=np.float32)
    delta = seeds - ref[None, :]
    index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
    seed = seeds[index]
    return Vector((float(seed[0]), float(seed[1]), float(seed[2])))


def _body_seed_outside_test(point: np.ndarray,
                            ctx: _BodyRayContext,
                            seed_world: Vector) -> bool:
    p_world = Vector((float(point[0]), float(point[1]), float(point[2])))
    p_local = ctx.matrix_inv @ p_world
    seed_local = ctx.matrix_inv @ seed_world
    local_delta = p_local - seed_local
    local_dist = float(local_delta.length)
    if local_dist <= 1.0e-8:
        return False
    local_dir = local_delta / local_dist
    hit, _loc, _normal, _face = ctx.evaluated.ray_cast(
        seed_local,
        local_dir,
        distance=local_dist + 1.0e-6,
    )
    return bool(hit)


def _body_seed_escape_point(point: np.ndarray,
                            ctx: _BodyRayContext,
                            seed_world: Vector) -> tuple[np.ndarray, bool]:
    p_world = Vector((float(point[0]), float(point[1]), float(point[2])))
    p_local = ctx.matrix_inv @ p_world
    seed_local = ctx.matrix_inv @ seed_world
    local_delta = p_local - seed_local
    local_dist = float(local_delta.length)
    if local_dist <= 1.0e-8:
        return np.asarray(point, dtype=np.float32), False
    local_dir = local_delta / local_dist
    far_dist = max(local_dist + ctx.body_span * 2.0, ctx.body_span * 3.0)
    hit, loc, normal, _face = ctx.evaluated.ray_cast(seed_local, local_dir, distance=far_dist)
    if not hit:
        return np.asarray(point, dtype=np.float32), False
    loc_world = ctx.matrix @ loc
    normal_world = (ctx.normal_matrix @ normal).normalized()
    direction_world = p_world - seed_world
    if direction_world.length > 1.0e-8:
        direction_world.normalize()
        if normal_world.dot(direction_world) < 0.0:
            normal_world.negate()
    corrected = loc_world + normal_world * ctx.margin
    return _vector_np(corrected), True


def _valid_body_segment(prev: np.ndarray,
                        candidate: np.ndarray,
                        ctx: _BodyRayContext,
                        seed_world: Vector) -> bool:
    if not _body_seed_outside_test(candidate, ctx, seed_world):
        return False
    p0 = Vector((float(prev[0]), float(prev[1]), float(prev[2])))
    p1 = Vector((float(candidate[0]), float(candidate[1]), float(candidate[2])))
    p0_local = ctx.matrix_inv @ p0
    p1_local = ctx.matrix_inv @ p1
    delta = p1_local - p0_local
    distance = float(delta.length)
    if distance <= 1.0e-8:
        return True
    direction = delta / distance
    eps = max(1.0e-5, ctx.margin * 0.25)
    hit, loc, _normal, _face = ctx.evaluated.ray_cast(p0_local, direction, distance=distance)
    if not hit:
        return True
    hit_distance = float((loc - p0_local).length)
    if eps < hit_distance < distance - eps:
        return False
    if hit_distance <= eps and distance > eps * 2.0:
        origin = p0_local + direction * eps
        hit2, loc2, _normal2, _face2 = ctx.evaluated.ray_cast(origin, direction, distance=distance - eps)
        if hit2:
            hit2_distance = eps + float((loc2 - origin).length)
            if eps < hit2_distance < distance - eps:
                return False
    return True


def _repair_strand_fk_body(strand: np.ndarray,
                           rest_lengths: np.ndarray,
                           fallback_dirs: np.ndarray,
                           locked: int,
                           ctx: _BodyRayContext,
                           seed_world: Vector) -> tuple[np.ndarray, BodyFkRepairStats, np.ndarray]:
    pps = int(strand.shape[0])
    locked = max(1, min(int(locked), pps))
    repaired = np.asarray(strand, dtype=np.float32).copy()
    point_mask = np.zeros(pps, dtype=np.int32)
    stats = BodyFkRepairStats()

    for i in range(locked, pps):
        prev = repaired[i - 1]
        if i >= 2:
            prevprev = repaired[i - 2]
            straight_fallback = prev - prevprev
        else:
            straight_fallback = np.asarray(fallback_dirs[max(i - 1, 0)], dtype=np.float32)
            prevprev = prev - straight_fallback
        segment_index = max(0, min(i - 1, len(rest_lengths) - 1))
        length = max(float(rest_lengths[segment_index]), 1.0e-8)
        fallback_dir = np.asarray(fallback_dirs[segment_index], dtype=np.float32)
        straight_dir = _normalize_np(prev - prevprev, fallback_dir)
        xpbd_dir = _normalize_np(np.asarray(strand[i], dtype=np.float32) - prev, straight_dir)
        candidate = (prev + xpbd_dir * length).astype(np.float32, copy=False)

        if _valid_body_segment(prev, candidate, ctx, seed_world):
            repaired[i] = candidate
            continue

        found = None
        lower_t = 0.0
        upper_t = None
        for t in BODY_FK_SEARCH_T_VALUES:
            direction = _slerp_direction_np(xpbd_dir, straight_dir, t)
            test = (prev + direction * length).astype(np.float32, copy=False)
            if _valid_body_segment(prev, test, ctx, seed_world):
                upper_t = float(t)
                break
            lower_t = float(t)

        if upper_t is not None:
            hi = upper_t
            lo = lower_t
            found = (prev + _slerp_direction_np(xpbd_dir, straight_dir, hi) * length).astype(np.float32, copy=False)
            for _step in range(BODY_FK_BINARY_STEPS):
                mid = (lo + hi) * 0.5
                direction = _slerp_direction_np(xpbd_dir, straight_dir, mid)
                test = (prev + direction * length).astype(np.float32, copy=False)
                if _valid_body_segment(prev, test, ctx, seed_world):
                    hi = mid
                    found = test
                else:
                    lo = mid

        if found is not None:
            repaired[i] = found
            point_mask[i] = 1
            stats.points += 1
            continue

        escape, escaped = _body_seed_escape_point(candidate, ctx, seed_world)
        if escaped:
            escape_dir = _normalize_np(escape - prev, straight_dir)
            length_candidate = (prev + escape_dir * length).astype(np.float32, copy=False)
            if _valid_body_segment(prev, length_candidate, ctx, seed_world):
                repaired[i] = length_candidate
            else:
                repaired[i] = escape
                if (
                    not _body_seed_outside_test(escape, ctx, seed_world)
                    or not _valid_body_segment(prev, escape, ctx, seed_world)
                ):
                    stats.failed_points += 1
            point_mask[i] = 1
            stats.points += 1
            stats.escape_points += 1
        else:
            repaired[i] = candidate
            point_mask[i] = 1
            stats.points += 1
            stats.failed_points += 1

    if np.any(point_mask):
        stats.strands = 1
    return repaired, stats, point_mask


def _apply_body_fk_hard_repair(points: np.ndarray,
                               pps: int,
                               rest_lengths: np.ndarray | None,
                               fallback_dirs: np.ndarray | None,
                               body_obj,
                               margin: float,
                               seed_world=None,
                               root_locked_points: int = 1,
                               active_strands: np.ndarray | None = None
                               ) -> tuple[np.ndarray, BodyFkRepairStats, np.ndarray]:
    if body_obj is None or body_obj.type != "MESH":
        n_strands = int(len(points) // max(int(pps), 1))
        return points, BodyFkRepairStats(), np.zeros(n_strands, dtype=np.int32)
    pps = int(pps)
    if pps < 2:
        return points, BodyFkRepairStats(), np.zeros(0, dtype=np.int32)
    strands = np.asarray(points, dtype=np.float32).reshape(-1, pps, 3)
    n_strands = int(strands.shape[0])
    if rest_lengths is None or fallback_dirs is None:
        return points, BodyFkRepairStats(), np.zeros(n_strands, dtype=np.int32)
    rest = np.asarray(rest_lengths, dtype=np.float32)
    fallback = np.asarray(fallback_dirs, dtype=np.float32)
    if rest.shape != (n_strands, pps - 1) or fallback.shape != (n_strands, pps - 1, 3):
        return points, BodyFkRepairStats(), np.zeros(n_strands, dtype=np.int32)
    seeds = _body_internal_seed_points(body_obj, preferred_seed=seed_world)
    if len(seeds) == 0:
        return points, BodyFkRepairStats(), np.zeros(n_strands, dtype=np.int32)
    ctx = _body_ray_context(body_obj, margin)
    if ctx is None:
        return points, BodyFkRepairStats(), np.zeros(n_strands, dtype=np.int32)

    if active_strands is None:
        active = np.ones(n_strands, dtype=bool)
    else:
        active = np.asarray(active_strands, dtype=bool).reshape(-1)
        if active.shape != (n_strands,):
            raise ValueError(f"body FK active shape mismatch: {active.shape} != {(n_strands,)}")
    if not np.any(active):
        return points, BodyFkRepairStats(), np.zeros(n_strands, dtype=np.int32)

    out = strands.copy()
    repaired_strands = np.zeros(n_strands, dtype=np.int32)
    total = BodyFkRepairStats()
    for strand_index in np.flatnonzero(active):
        strand = out[int(strand_index)]
        seed = _choose_body_seed(strand[-1], strand[0], seeds)
        if seed is None:
            continue
        repaired, stats, point_mask = _repair_strand_fk_body(
            strand,
            rest[int(strand_index)],
            fallback[int(strand_index)],
            int(root_locked_points),
            ctx,
            seed,
        )
        if stats.points > 0:
            out[int(strand_index)] = repaired
            repaired_strands[int(strand_index)] = 1
            total.strands += stats.strands
            total.points += stats.points
            total.escape_points += stats.escape_points
            total.failed_points += stats.failed_points
    return np.ascontiguousarray(out.reshape(-1, 3), dtype=np.float32), total, repaired_strands


def _apply_body_seed_hard_guard(points: np.ndarray, body_obj,
                                margin: float,
                                seed_world=None,
                                points_per_strand: int | None = None):
    pps_for_mask = int(points_per_strand) if points_per_strand is not None else 0
    empty_strands = np.zeros(
        int(math.ceil(len(points) / float(pps_for_mask))) if pps_for_mask > 0 else 0,
        dtype=np.int32,
    )
    if body_obj is None or body_obj.type != "MESH":
        if points_per_strand is None:
            return points, 0
        return points, 0, empty_strands
    if seed_world is None:
        seed_world = _head_seed_world(body_obj)
    if seed_world is None:
        if points_per_strand is None:
            return points, 0
        return points, 0, empty_strands

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = body_obj.evaluated_get(depsgraph)
    matrix = evaluated.matrix_world.copy()
    matrix_inv = matrix.inverted()
    seed_local = matrix_inv @ Vector(seed_world)
    dims = getattr(body_obj, "dimensions", (1.0, 1.0, 1.0))
    body_span = max(float(max(dims)), 0.5)
    margin = max(float(margin), 0.0)
    out = np.asarray(points, dtype=np.float32).copy()
    fixed = 0
    fixed_strands = None
    pps = pps_for_mask
    if pps > 0:
        fixed_strands = empty_strands

    for index, point in enumerate(out):
        p_world = Vector((float(point[0]), float(point[1]), float(point[2])))
        p_local = matrix_inv @ p_world
        local_delta = p_local - seed_local
        local_dist = float(local_delta.length)
        if local_dist <= 1.0e-8:
            continue
        local_dir = local_delta / local_dist
        hit, _loc, _normal, _face = evaluated.ray_cast(seed_local, local_dir, distance=local_dist)
        if hit:
            continue

        far_dist = max(local_dist + body_span * 2.0, body_span * 3.0)
        hit, loc, normal, _face = evaluated.ray_cast(seed_local, local_dir, distance=far_dist)
        if not hit:
            continue

        loc_world = matrix @ loc
        normal_world = (matrix.to_3x3() @ normal).normalized()
        direction_world = (p_world - Vector(seed_world)).normalized()
        if normal_world.dot(direction_world) < 0.0:
            normal_world.negate()
        corrected = loc_world + normal_world * margin
        out[index] = (corrected.x, corrected.y, corrected.z)
        fixed += 1
        if fixed_strands is not None:
            fixed_strands[int(index // pps)] = 1
    if fixed_strands is None:
        return np.ascontiguousarray(out, dtype=np.float32), fixed
    return np.ascontiguousarray(out, dtype=np.float32), fixed, fixed_strands


def _rest_lengths(points: np.ndarray, pps: int) -> tuple[np.ndarray, np.ndarray]:
    strands = points.reshape(-1, pps, 3)
    seg = np.linalg.norm(strands[:, 1:, :] - strands[:, :-1, :], axis=2)
    bend = np.linalg.norm(strands[:, 2:, :] - strands[:, :-2, :], axis=2)
    return (
        np.ascontiguousarray(np.maximum(seg, 1.0e-6).reshape(-1), dtype=np.float32),
        np.ascontiguousarray(np.maximum(bend, 1.0e-6).reshape(-1), dtype=np.float32),
    )


def _segment_lengths_and_dirs(points: np.ndarray, pps: int) -> tuple[np.ndarray, np.ndarray]:
    strands = points.reshape(-1, int(pps), 3)
    seg = strands[:, 1:, :] - strands[:, :-1, :]
    lengths = np.linalg.norm(seg, axis=2).astype(np.float32)
    fallback = np.zeros_like(seg, dtype=np.float32)
    fallback[:, :, 2] = -1.0
    valid = lengths > 1.0e-8
    dirs = fallback
    dirs[valid] = (seg[valid] / lengths[valid][:, None]).astype(np.float32)
    return (
        np.ascontiguousarray(np.maximum(lengths, 1.0e-8), dtype=np.float32),
        np.ascontiguousarray(dirs, dtype=np.float32),
    )


def _keep_length_fk(points: np.ndarray, pps: int,
                    rest_lengths: np.ndarray,
                    fallback_dirs: np.ndarray) -> tuple[np.ndarray, float]:
    strands = np.asarray(points, dtype=np.float32).reshape(-1, int(pps), 3)
    if rest_lengths.shape != (strands.shape[0], int(pps) - 1):
        raise ValueError(
            "Keep Length rest length shape mismatch: "
            f"{rest_lengths.shape} != {(strands.shape[0], int(pps) - 1)}"
        )
    seg = strands[:, 1:, :] - strands[:, :-1, :]
    seg_lengths = np.linalg.norm(seg, axis=2).astype(np.float32)
    valid = seg_lengths > 1.0e-8
    dirs = np.asarray(fallback_dirs, dtype=np.float32).copy()
    dirs[valid] = (seg[valid] / seg_lengths[valid][:, None]).astype(np.float32)
    scaled = dirs * rest_lengths[:, :, None]
    out = np.empty_like(strands, dtype=np.float32)
    out[:, 0, :] = strands[:, 0, :]
    out[:, 1:, :] = out[:, 0:1, :] + np.cumsum(scaled, axis=1, dtype=np.float32)
    out_seg = out[:, 1:, :] - out[:, :-1, :]
    out_lengths = np.linalg.norm(out_seg, axis=2)
    max_error = float(np.max(np.abs(out_lengths - rest_lengths)) * 1000.0) if out_lengths.size else 0.0
    return np.ascontiguousarray(out.reshape(-1, 3), dtype=np.float32), max_error


def _inverse_mass(n_strands: int, pps: int, root_locked_points: int) -> np.ndarray:
    locked = max(1, min(int(root_locked_points), pps))
    inv = np.ones(n_strands * pps, dtype=np.float32)
    for strand in range(n_strands):
        base = strand * pps
        inv[base:base + locked] = 0.0
    return inv


def _guide_point_indices(guide_indices: np.ndarray, pps: int) -> np.ndarray:
    local = np.arange(int(pps), dtype=np.int64)
    return (guide_indices.astype(np.int64)[:, None] * int(pps) + local[None, :]).reshape(-1)


def _decimated_guide_indices(n_strands: int, guide_decimation: int) -> np.ndarray:
    decimation = max(1, int(guide_decimation))
    if decimation <= 1 or n_strands <= 1:
        return np.arange(n_strands, dtype=np.int64)
    return np.arange(0, n_strands, decimation, dtype=np.int64)


def _full_active_strands_from_guides(n_strands: int,
                                     guide_indices: np.ndarray,
                                     guide_active: np.ndarray,
                                     guide_nearest: np.ndarray | None) -> np.ndarray:
    active = np.zeros(int(n_strands), dtype=np.int32)
    guide_active = np.asarray(guide_active, dtype=bool).reshape(-1)
    guide_indices = np.asarray(guide_indices, dtype=np.int64).reshape(-1)
    if guide_active.shape != guide_indices.shape:
        raise ValueError(f"guide active shape mismatch: {guide_active.shape} != {guide_indices.shape}")
    if np.any(guide_active):
        active[guide_indices[guide_active]] = 1
        if guide_nearest is not None:
            nearest = np.asarray(guide_nearest, dtype=np.int64)
            active[np.any(guide_active[nearest], axis=1)] = 1
    return active


def _guide_interpolation_weights(init_world: np.ndarray, pps: int,
                                 guide_indices: np.ndarray,
                                 max_guides: int = 4) -> tuple[np.ndarray, np.ndarray]:
    n_strands = int(len(init_world) // pps)
    guide_count = int(len(guide_indices))
    if guide_count <= 0:
        raise ValueError("Guide decimation left no strands to simulate")
    k = max(1, min(int(max_guides), guide_count))
    roots = init_world.reshape(n_strands, pps, 3)[:, 0, :]
    guide_roots = roots[guide_indices]
    nearest = np.empty((n_strands, k), dtype=np.int64)
    weights = np.empty((n_strands, k), dtype=np.float32)
    chunk_size = 4096
    for start in range(0, n_strands, chunk_size):
        end = min(start + chunk_size, n_strands)
        d = roots[start:end, None, :] - guide_roots[None, :, :]
        d2 = np.einsum("cgj,cgj->cg", d, d, optimize=True)
        if k == guide_count:
            order = np.argsort(d2, axis=1)[:, :k]
        else:
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            order = np.take_along_axis(
                part,
                np.argsort(np.take_along_axis(d2, part, axis=1), axis=1),
                axis=1,
            )
        selected_d2 = np.take_along_axis(d2, order, axis=1)
        near_weights = np.empty_like(selected_d2, dtype=np.float32)
        exact = selected_d2[:, 0] <= 1.0e-14
        if np.any(exact):
            near_weights[exact] = 0.0
            near_weights[exact, 0] = 1.0
        if np.any(~exact):
            inv = 1.0 / np.maximum(selected_d2[~exact], 1.0e-12)
            near_weights[~exact] = (inv / np.sum(inv, axis=1, keepdims=True)).astype(np.float32)
        nearest[start:end] = order
        weights[start:end] = near_weights
    for local_guide, strand_index in enumerate(guide_indices.tolist()):
        nearest[int(strand_index), :] = local_guide
        weights[int(strand_index), :] = 0.0
        weights[int(strand_index), 0] = 1.0
    return nearest, weights


def _restore_decimated_strands(eval_world: np.ndarray,
                               guide_eval_world: np.ndarray,
                               guide_sim_world: np.ndarray,
                               pps: int,
                               nearest: np.ndarray,
                               weights: np.ndarray) -> np.ndarray:
    n_strands = int(len(eval_world) // pps)
    eval_strands = eval_world.reshape(n_strands, pps, 3)
    guide_count = int(len(guide_sim_world) // pps)
    guide_delta = (
        guide_sim_world.reshape(guide_count, pps, 3)
        - guide_eval_world.reshape(guide_count, pps, 3)
    )
    blended_delta = np.sum(
        guide_delta[nearest] * weights[:, :, None, None],
        axis=1,
        dtype=np.float32,
    )
    return np.ascontiguousarray((eval_strands + blended_delta).reshape(-1, 3), dtype=np.float32)


def _target_motion_mm(prev_targets: np.ndarray, next_targets: np.ndarray,
                      inv_mass: np.ndarray) -> float:
    locked = inv_mass <= 0.0
    if not np.any(locked):
        return 0.0
    delta = next_targets[locked] - prev_targets[locked]
    return float(np.max(np.linalg.norm(delta, axis=1)) * 1000.0) if len(delta) else 0.0


def _read_targets_for_frames(curves_obj, frames: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    target_worlds: dict[int, np.ndarray] = {}
    offsets: dict[int, np.ndarray] = {}
    expected_points = None
    scene = bpy.context.scene
    for frame in frames:
        scene.frame_set(int(frame))
        eval_world, original_world = _read_world(curves_obj)
        if expected_points is None:
            expected_points = len(eval_world)
        elif len(eval_world) != expected_points:
            raise ValueError(
                "Curves point count changed during simulation target read: "
                f"frame {frame} has {len(eval_world)} points, expected {expected_points}"
            )
        target_worlds[int(frame)] = eval_world.copy()
        offsets[int(frame)] = eval_world - original_world
    return target_worlds, offsets


def _bake_position_keyframes(curves_obj, frames: list[int],
                             baked: dict[int, np.ndarray],
                             offsets: dict[int, np.ndarray],
                             pps: int | None = None,
                             keep_rest_lengths: np.ndarray | None = None,
                             keep_fallback_dirs: np.ndarray | None = None,
                             body_hard_guard_obj=None,
                             body_hard_guard_margin: float = 0.0,
                             body_hard_guard_seed=None,
                             body_fk_root_locked_points: int = 1) -> None:
    local_values = _local_values_from_world(
        curves_obj,
        frames,
        baked,
        offsets,
        pps=pps,
        keep_rest_lengths=keep_rest_lengths,
        keep_fallback_dirs=keep_fallback_dirs,
        body_hard_guard_obj=body_hard_guard_obj,
        body_hard_guard_margin=body_hard_guard_margin,
        body_hard_guard_seed=body_hard_guard_seed,
        body_fk_root_locked_points=body_fk_root_locked_points,
    )
    _bake_local_position_keyframes(curves_obj, frames, local_values)


def _local_values_from_world(curves_obj, frames: list[int],
                             baked: dict[int, np.ndarray],
                             offsets: dict[int, np.ndarray],
                             pps: int | None = None,
                             keep_rest_lengths: np.ndarray | None = None,
                             keep_fallback_dirs: np.ndarray | None = None,
                             body_hard_guard_obj=None,
                             body_hard_guard_margin: float = 0.0,
                             body_hard_guard_seed=None,
                             body_fk_root_locked_points: int = 1,
                             correction_passes: int = 3) -> np.ndarray:
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    n_total = len(attr.data)
    if n_total == 0:
        raise ValueError("Curves has no points")

    scene = bpy.context.scene
    local_values = np.empty((len(frames), n_total, 3), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        scene.frame_set(frame)
        local = _world_to_local_points(
            curves_obj,
            baked[frame],
            offset=offsets[frame],
        )
        if pps is not None and keep_rest_lengths is not None and keep_fallback_dirs is not None:
            for _ in range(max(1, int(correction_passes))):
                _write_local_points(curves_obj, local)
                bpy.context.view_layer.update()
                eval_world, original_world = _read_world(curves_obj)
                corrected, _keep_error = _keep_length_fk(
                    eval_world,
                    int(pps),
                    keep_rest_lengths,
                    keep_fallback_dirs,
                )
                local = _world_to_local_points(
                    curves_obj,
                    corrected,
                    offset=eval_world - original_world,
                )
        if body_hard_guard_obj is not None and body_hard_guard_seed is not None:
            _write_local_points(curves_obj, local)
            bpy.context.view_layer.update()
            eval_world, original_world = _read_world(curves_obj)
            if pps is not None:
                guarded, fixed, guard_strands = _apply_body_seed_hard_guard(
                    eval_world,
                    body_hard_guard_obj,
                    body_hard_guard_margin,
                    body_hard_guard_seed,
                    points_per_strand=int(pps),
                )
                if fixed and keep_rest_lengths is not None and keep_fallback_dirs is not None:
                    guarded, _repair_stats, _repair_strands = _apply_body_fk_hard_repair(
                        guarded,
                        int(pps),
                        keep_rest_lengths,
                        keep_fallback_dirs,
                        body_hard_guard_obj,
                        body_hard_guard_margin,
                        body_hard_guard_seed,
                        root_locked_points=int(body_fk_root_locked_points),
                        active_strands=guard_strands,
                    )
            else:
                guarded, _fixed = _apply_body_seed_hard_guard(
                    eval_world,
                    body_hard_guard_obj,
                    body_hard_guard_margin,
                    body_hard_guard_seed,
                )
            local = _world_to_local_points(
                curves_obj,
                guarded,
                offset=eval_world - original_world,
            )
        local_values[frame_index] = local
    return local_values


def _bake_local_position_keyframes(curves_obj, frames: list[int],
                                   local_values: np.ndarray) -> None:
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    n_total = len(attr.data)
    if n_total == 0:
        raise ValueError("Curves has no points")
    if local_values.shape != (len(frames), n_total, 3):
        raise ValueError(
            "Cached position shape mismatch: "
            f"{local_values.shape} != {(len(frames), n_total, 3)}"
        )
    data = curves_obj.data
    anim = data.animation_data_create()
    if anim.action is None:
        anim.action = bpy.data.actions.new(f"Yurameki Bake {curves_obj.name}")
    action = anim.action

    scene = bpy.context.scene
    frame_numbers = np.asarray(frames, dtype=np.float32)
    co = np.empty(len(frames) * 2, dtype=np.float32)
    co[0::2] = frame_numbers
    interpolation = np.ones(len(frames), dtype=np.int32)
    for point_index in range(n_total):
        if point_index and point_index % 10000 == 0:
            print(f"Yurameki Bake keyframes point={point_index}/{n_total}")
        data_path = f'attributes["position"].data[{point_index}].vector'
        for axis in range(3):
            fcurve = action.fcurve_ensure_for_datablock(
                data,
                data_path,
                index=axis,
                group_name="Yurameki Position",
            )
            fcurve.keyframe_points.clear()
            fcurve.keyframe_points.add(len(frames))
            co[1::2] = local_values[:, point_index, axis]
            fcurve.keyframe_points.foreach_set("co", co)
            fcurve.keyframe_points.foreach_set("interpolation", interpolation)
            fcurve.update()

    scene.frame_set(frames[-1])
    bpy.context.view_layer.update()


def _cache_dir() -> str:
    blend_path = bpy.data.filepath
    if blend_path:
        path = bpy.path.abspath("//yurameki_cache")
    else:
        path = os.path.join(tempfile.gettempdir(), "yurameki_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_file_path(curves_obj, start_frame: int, end_frame: int) -> str:
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in curves_obj.name)
    return os.path.join(_cache_dir(), f"{safe_name}_{start_frame}_{end_frame}.npz")


def _write_local_points(curves_obj, local_pts: np.ndarray) -> None:
    attr = curves_obj.data.attributes.get("position")
    if attr is None or len(attr.data) != len(local_pts):
        raise ValueError("Curves position attribute shape changed")
    attr.data.foreach_set("vector", np.ascontiguousarray(local_pts, dtype=np.float32).ravel())
    curves_obj.data.update_tag()


def _apply_cache_frame(cache: YuramekiRuntimeCache, frame: int) -> bool:
    matches = np.nonzero(cache.frames == int(frame))[0]
    if len(matches) == 0:
        return False
    obj = bpy.data.objects.get(cache.object_name)
    if obj is None or obj.type != "CURVES" or obj.data.name != cache.data_name:
        return False
    _write_local_points(obj, cache.local_values[int(matches[0])])
    return True


def _register_sim_cache(curves_obj, frames: list[int],
                        baked: dict[int, np.ndarray],
                        offsets: dict[int, np.ndarray],
                        pps: int | None = None,
                        keep_rest_lengths: np.ndarray | None = None,
                        keep_fallback_dirs: np.ndarray | None = None,
                        body_hard_guard_obj=None,
                        body_hard_guard_margin: float = 0.0,
                        body_hard_guard_seed=None,
                        body_fk_root_locked_points: int = 1) -> YuramekiRuntimeCache:
    local_values = _local_values_from_world(
        curves_obj,
        frames,
        baked,
        offsets,
        pps=pps,
        keep_rest_lengths=keep_rest_lengths,
        keep_fallback_dirs=keep_fallback_dirs,
        body_hard_guard_obj=body_hard_guard_obj,
        body_hard_guard_margin=body_hard_guard_margin,
        body_hard_guard_seed=body_hard_guard_seed,
        body_fk_root_locked_points=body_fk_root_locked_points,
    )
    frame_array = np.asarray(frames, dtype=np.int32)
    path = _cache_file_path(curves_obj, int(frame_array[0]), int(frame_array[-1]))
    np.savez(
        path,
        frames=frame_array,
        local_values=np.ascontiguousarray(local_values, dtype=np.float32),
        object_name=np.array([curves_obj.name]),
        data_name=np.array([curves_obj.data.name]),
    )
    cache = YuramekiRuntimeCache(
        object_name=curves_obj.name,
        data_name=curves_obj.data.name,
        frames=frame_array,
        local_values=np.ascontiguousarray(local_values, dtype=np.float32),
        path=os.path.abspath(path),
    )
    _CACHE_REGISTRY[curves_obj.name] = cache
    curves_obj.data["yurameki_cache_path"] = cache.path
    curves_obj.data["yurameki_cache_start"] = cache.start_frame
    curves_obj.data["yurameki_cache_end"] = cache.end_frame
    register_cache_handler()
    _apply_cache_frame(cache, int(bpy.context.scene.frame_current))
    return cache


def clear_runtime_cache(curves_obj) -> None:
    if curves_obj is None:
        return
    _CACHE_REGISTRY.pop(curves_obj.name, None)
    for key in ("yurameki_cache_path", "yurameki_cache_start", "yurameki_cache_end"):
        try:
            del curves_obj.data[key]
        except Exception:
            pass
    if not _CACHE_REGISTRY:
        unregister_cache_handler()


def get_runtime_cache(curves_obj) -> YuramekiRuntimeCache | None:
    if curves_obj is None:
        return None
    cache = _CACHE_REGISTRY.get(curves_obj.name)
    if cache is not None and cache.data_name == curves_obj.data.name:
        return cache
    path = str(curves_obj.data.get("yurameki_cache_path", "")).strip()
    if not path or not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            frames = np.ascontiguousarray(data["frames"], dtype=np.int32)
            local_values = np.ascontiguousarray(data["local_values"], dtype=np.float32)
    except Exception:
        return None
    if local_values.ndim != 3 or local_values.shape[2] != 3:
        return None
    attr = curves_obj.data.attributes.get("position")
    if attr is None or local_values.shape[1] != len(attr.data):
        return None
    cache = YuramekiRuntimeCache(
        object_name=curves_obj.name,
        data_name=curves_obj.data.name,
        frames=frames,
        local_values=local_values,
        path=os.path.abspath(path),
    )
    _CACHE_REGISTRY[curves_obj.name] = cache
    register_cache_handler()
    return cache


def bake_runtime_cache(curves_obj) -> dict:
    cache = get_runtime_cache(curves_obj)
    if cache is None:
        raise ValueError("No Yurameki runtime cache for this Curves object")
    _bake_local_position_keyframes(
        curves_obj,
        [int(frame) for frame in cache.frames.tolist()],
        cache.local_values,
    )
    return {
        "object": cache.object_name,
        "start_frame": cache.start_frame,
        "end_frame": cache.end_frame,
        "n_frames": int(len(cache.frames)),
        "n_points": cache.n_points,
        "fcurves": cache.n_points * 3,
        "keys": cache.n_points * 3 * int(len(cache.frames)),
        "path": cache.path,
    }


@persistent
def _yurameki_cache_frame_change(_scene):
    if _CACHE_MUTED:
        return
    frame = int(bpy.context.scene.frame_current)
    for cache in tuple(_CACHE_REGISTRY.values()):
        _apply_cache_frame(cache, frame)


def register_cache_handler() -> None:
    handlers = bpy.app.handlers.frame_change_post
    if _yurameki_cache_frame_change not in handlers:
        handlers.append(_yurameki_cache_frame_change)


def unregister_cache_handler() -> None:
    handlers = bpy.app.handlers.frame_change_post
    if _yurameki_cache_frame_change in handlers:
        handlers.remove(_yurameki_cache_frame_change)


class WarpJointSimulator:
    def __init__(
        self,
        init_positions: np.ndarray,
        points_per_strand: int,
        root_locked_points: int,
        particle_mass: float,
        device: str = "cuda:0",
    ):
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("NVIDIA Warp CUDA device is unavailable")
        self.device = str(device)
        device_info = wp.get_device(self.device)
        if not bool(getattr(device_info, "is_cuda", False)):
            raise RuntimeError(f"Warp device is not CUDA: {device_info}")
        wp.set_device(self.device)
        self.device_name = str(getattr(device_info, "name", self.device))
        self.device_arch = int(getattr(device_info, "arch", 0) or 0)
        self.pps = int(points_per_strand)
        self.n_total = int(len(init_positions))
        self.n_strands = self.n_total // self.pps
        self.n_segments = self.n_strands * (self.pps - 1)
        self.n_bends = self.n_strands * max(self.pps - 2, 0)
        self.inv_mass_np = _inverse_mass(self.n_strands, self.pps, root_locked_points)
        self.seg_rest_np, self.bend_rest_np = _rest_lengths(init_positions, self.pps)

        positions = np.ascontiguousarray(init_positions, dtype=np.float32)
        inv_mass = self.inv_mass_np / max(float(particle_mass), 1.0e-8)
        inv_mass[self.inv_mass_np <= 0.0] = 0.0
        self.pos = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.vel = wp.zeros(self.n_total, dtype=wp.vec3, device=self.device)
        self.predicted = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.target_start = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.target_end = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.inv_mass = wp.array(inv_mass, dtype=float, device=self.device)
        self.segment_rest = wp.array(self.seg_rest_np, dtype=float, device=self.device)
        self.bend_rest = wp.array(self.bend_rest_np, dtype=float, device=self.device)
        self.contact_mask = wp.zeros(self.n_total, dtype=wp.int32, device=self.device)
        self.frame_contact_strands = wp.zeros(self.n_strands, dtype=wp.int32, device=self.device)
        self.post_active_strands = wp.zeros(self.n_strands, dtype=wp.int32, device=self.device)
        self.post_contact_strands = wp.zeros(self.n_strands, dtype=wp.int32, device=self.device)
        self.keep_segment_rest = wp.array(self.seg_rest_np, dtype=float, device=self.device)
        self.hit_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._last_meshes: ColliderMeshSet | None = None

    def set_targets(self, start: np.ndarray, end: np.ndarray) -> None:
        self.target_start.assign(np.ascontiguousarray(start, dtype=np.float32))
        self.target_end.assign(np.ascontiguousarray(end, dtype=np.float32))

    def set_positions(self, positions: np.ndarray) -> None:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.shape != (self.n_total, 3):
            raise ValueError(f"position shape mismatch: {positions.shape} != {(self.n_total, 3)}")
        self.pos.assign(positions)
        self.predicted.assign(positions)

    def set_keep_rest_lengths(self, rest_lengths: np.ndarray) -> None:
        rest = np.ascontiguousarray(rest_lengths.reshape(-1), dtype=np.float32)
        if rest.shape != (self.n_segments,):
            raise ValueError(f"keep rest length shape mismatch: {rest.shape} != {(self.n_segments,)}")
        self.keep_segment_rest.assign(rest)

    def zero_strand_velocities(self, active_strands: np.ndarray) -> int:
        active = np.ascontiguousarray(active_strands.astype(np.int32, copy=False).reshape(-1))
        if active.shape != (self.n_strands,):
            raise ValueError(f"zero velocity strand shape mismatch: {active.shape} != {(self.n_strands,)}")
        if not np.any(active):
            return 0
        self.post_contact_strands.assign(active)
        wp.launch(
            _zero_contact_strand_velocity_kernel,
            dim=self.n_total,
            inputs=[self.vel, self.post_contact_strands, self.pps],
            device=self.device,
        )
        wp.synchronize()
        return int(np.count_nonzero(active) * self.pps)

    def _make_mesh(self, collider_objects, support_winding_number: bool = False) -> tuple[wp.Mesh, int, int]:
        vertices, indices = _evaluated_mesh_arrays(collider_objects)
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=self.device),
            indices=wp.array(indices, dtype=wp.int32, device=self.device),
            support_winding_number=bool(support_winding_number),
        )
        return mesh, int(len(vertices)), int(len(indices) // 3)

    def make_meshes(self, collider_objects) -> ColliderMeshSet:
        if not collider_objects:
            raise ValueError("expected at least one Mesh collider")
        body_mesh, body_vertices, body_triangles = self._make_mesh([collider_objects[0]])
        clothes_mesh = None
        clothes_vertices = 0
        clothes_triangles = 0
        clothes_objects = list(collider_objects[1:])
        if clothes_objects:
            clothes_mesh, clothes_vertices, clothes_triangles = self._make_mesh(clothes_objects)
        return ColliderMeshSet(
            body=body_mesh,
            clothes=clothes_mesh,
            n_vertices=body_vertices + clothes_vertices,
            n_triangles=body_triangles + clothes_triangles,
        )

    def estimated_free_move_mm(self, dt_frame: float,
                               gravity: tuple[float, float, float]) -> float:
        free = self.inv_mass_np > 0.0
        if not np.any(free):
            return 0.0
        velocity = self.vel.numpy().astype(np.float32, copy=False)
        gravity_vec = np.asarray(gravity, dtype=np.float32)
        motion = velocity[free] * float(dt_frame) + gravity_vec * (0.5 * float(dt_frame) * float(dt_frame))
        return float(np.max(np.linalg.norm(motion, axis=1)) * 1000.0) if len(motion) else 0.0

    def _solve_constraints(self, dt: float, iterations: int,
                           stretch_compliance: float,
                           bend_compliance: float) -> None:
        for _ in range(max(1, int(iterations))):
            for parity in (0, 1):
                wp.launch(
                    _solve_distance_kernel,
                    dim=self.n_segments,
                    inputs=[
                        self.predicted,
                        self.inv_mass,
                        self.segment_rest,
                        self.pps,
                        parity,
                        float(stretch_compliance),
                        float(dt),
                    ],
                    device=self.device,
                )
            if self.n_bends > 0 and bend_compliance >= 0.0:
                for parity in (0, 1):
                    wp.launch(
                        _solve_bend_kernel,
                        dim=self.n_bends,
                        inputs=[
                            self.predicted,
                            self.inv_mass,
                            self.bend_rest,
                            self.pps,
                            parity,
                            float(bend_compliance),
                            float(dt),
                        ],
                        device=self.device,
                    )

    def _collide(self, meshes: ColliderMeshSet, margin: float, search_distance: float,
                 max_correction: float, collision_response: float,
                 allow_sweep: bool, segment_passes: int) -> None:
        max_correction = max(float(max_correction), 1.0e-6)
        collision_response = min(max(float(collision_response), 0.0), 1.0)
        if meshes.body is not None:
            wp.launch(
                _body_point_collision_kernel,
                dim=self.n_total,
                inputs=[
                    meshes.body.id,
                    self.pos,
                    self.predicted,
                    self.vel,
                    self.inv_mass,
                    self.pps,
                    float(margin),
                    float(search_distance),
                    max_correction,
                    collision_response,
                    int(bool(allow_sweep)),
                    self.contact_mask,
                    self.frame_contact_strands,
                    self.hit_count,
                ],
                device=self.device,
            )
        if meshes.clothes is not None:
            wp.launch(
                _cloth_point_collision_kernel,
                dim=self.n_total,
                inputs=[
                    meshes.clothes.id,
                    self.pos,
                    self.predicted,
                    self.vel,
                    self.inv_mass,
                    self.pps,
                    float(margin),
                    float(search_distance),
                    max_correction,
                    collision_response,
                    int(bool(allow_sweep)),
                    self.contact_mask,
                    self.frame_contact_strands,
                    self.hit_count,
                ],
                device=self.device,
            )
        for _ in range(max(1, int(segment_passes))):
            for parity in (0, 1):
                if meshes.body is not None:
                    wp.launch(
                        _segment_collision_kernel,
                        dim=self.n_segments,
                        inputs=[
                            meshes.body.id,
                            self.predicted,
                            self.vel,
                            self.inv_mass,
                            self.pps,
                            float(margin),
                            max_correction,
                            collision_response,
                            parity,
                            self.contact_mask,
                            self.frame_contact_strands,
                            self.hit_count,
                        ],
                        device=self.device,
                    )
                if meshes.clothes is not None:
                    wp.launch(
                        _segment_collision_kernel,
                        dim=self.n_segments,
                        inputs=[
                            meshes.clothes.id,
                            self.predicted,
                            self.vel,
                            self.inv_mass,
                            self.pps,
                            float(margin),
                            max_correction,
                            collision_response,
                            parity,
                            self.contact_mask,
                            self.frame_contact_strands,
                            self.hit_count,
                        ],
                        device=self.device,
                    )

    def post_keep_length_collision(
        self,
        active_strands: np.ndarray,
        margin: float,
        search_distance: float,
        max_correction: float,
        collision_response: float,
        passes: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        meshes = self._last_meshes
        if meshes is None:
            return self.pos.numpy().astype(np.float32, copy=True), 0, np.zeros(self.n_strands, dtype=np.int32)

        active = np.ascontiguousarray(active_strands.astype(np.int32, copy=False).reshape(-1))
        if active.shape != (self.n_strands,):
            raise ValueError(f"active strand shape mismatch: {active.shape} != {(self.n_strands,)}")
        if not np.any(active):
            return self.pos.numpy().astype(np.float32, copy=True), 0, np.zeros(self.n_strands, dtype=np.int32)

        max_correction = max(float(max_correction), 1.0e-6)
        collision_response = min(max(float(collision_response), 0.0), 1.0)
        self.post_active_strands.assign(active)
        wp.launch(_clear_contact_mask_kernel, dim=self.n_strands, inputs=[self.post_contact_strands], device=self.device)
        self.hit_count.assign(np.zeros(1, dtype=np.int32))

        for _ in range(max(1, int(passes))):
            if meshes.body is not None:
                wp.launch(
                    _post_keep_body_collision_kernel,
                    dim=self.n_strands,
                    inputs=[
                        meshes.body.id,
                        self.pos,
                        self.inv_mass,
                        self.post_active_strands,
                        self.keep_segment_rest,
                        self.pps,
                        float(margin),
                        float(search_distance),
                        max_correction,
                        collision_response,
                        self.post_contact_strands,
                        self.hit_count,
                    ],
                    device=self.device,
                )
            if meshes.clothes is not None:
                wp.launch(
                    _post_keep_cloth_collision_kernel,
                    dim=self.n_strands,
                    inputs=[
                        meshes.clothes.id,
                        self.pos,
                        self.inv_mass,
                        self.post_active_strands,
                        self.keep_segment_rest,
                        self.pps,
                        float(margin),
                        float(search_distance),
                        max_correction,
                        collision_response,
                        self.post_contact_strands,
                        self.hit_count,
                    ],
                    device=self.device,
                )

        wp.launch(
            _zero_contact_strand_velocity_kernel,
            dim=self.n_total,
            inputs=[self.vel, self.post_contact_strands, self.pps],
            device=self.device,
        )
        wp.synchronize()
        hits = int(self.hit_count.numpy()[0])
        contacts = self.post_contact_strands.numpy().astype(np.int32, copy=True)
        return self.pos.numpy().astype(np.float32, copy=True), hits, contacts

    def simulate_frame(
        self,
        target_start: np.ndarray,
        target_end: np.ndarray,
        collider_objects,
        substeps: int,
        dt_frame: float,
        gravity: tuple[float, float, float],
        damping: float,
        max_velocity: float,
        iterations: int,
        stretch_compliance: float,
        bend_compliance: float,
        collision_margin: float,
        collision_search: float,
        collision_max_correction: float,
        collision_response: float,
        collision_velocity_damping: float,
        collision_passes: int,
        post_collision_iterations: int,
    ) -> tuple[np.ndarray, int, int, np.ndarray]:
        meshes = self.make_meshes(collider_objects)
        self._last_meshes = meshes
        self.set_targets(target_start, target_end)
        self.hit_count.assign(np.zeros(1, dtype=np.int32))
        wp.launch(
            _clear_contact_mask_kernel,
            dim=self.n_strands,
            inputs=[self.frame_contact_strands],
            device=self.device,
        )
        substeps = max(1, int(substeps))
        dt = float(dt_frame) / float(substeps)
        for step in range(substeps):
            alpha = float(step + 1) / float(substeps)
            wp.launch(
                _predict_kernel,
                dim=self.n_total,
                inputs=[
                    self.pos,
                    self.vel,
                    self.predicted,
                    self.target_start,
                    self.target_end,
                    self.inv_mass,
                    alpha,
                    dt,
                    float(gravity[0]),
                    float(gravity[1]),
                    float(gravity[2]),
                    float(max_velocity),
                ],
                device=self.device,
            )
            self._solve_constraints(dt, iterations, stretch_compliance, bend_compliance)
            wp.launch(
                _clear_contact_mask_kernel,
                dim=self.n_total,
                inputs=[self.contact_mask],
                device=self.device,
            )
            self._collide(meshes, collision_margin, collision_search,
                          collision_max_correction, collision_response,
                          True, collision_passes)
            for _ in range(max(0, int(post_collision_iterations))):
                self._solve_constraints(dt, 1, stretch_compliance, bend_compliance)
                self._collide(meshes, collision_margin, collision_search,
                              collision_max_correction, collision_response,
                              False, collision_passes)
            wp.launch(
                _derive_velocity_kernel,
                dim=self.n_total,
                inputs=[
                    self.pos,
                    self.predicted,
                    self.vel,
                    self.inv_mass,
                    self.contact_mask,
                    dt,
                    float(damping),
                    float(max_velocity),
                    min(max(float(collision_velocity_damping), 0.0), 1.0),
                ],
                device=self.device,
            )
            wp.launch(_commit_kernel, dim=self.n_total, inputs=[self.pos, self.predicted], device=self.device)
        wp.synchronize()
        hits = int(self.hit_count.numpy()[0])
        contact_strands = self.frame_contact_strands.numpy().astype(np.int32, copy=True)
        return self.pos.numpy().astype(np.float32, copy=True), hits, meshes.n_triangles, contact_strands


def check_warp_ready(curves_obj, collider_objects, root_locked_points: int,
                     particle_mass: float) -> WarpCheckStats:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    pps, n_strands = _uniform_points_per_strand(curves_obj)
    locked = max(1, min(int(root_locked_points), pps))
    world, _original = _read_world(curves_obj)
    sim = WarpJointSimulator(world, pps, locked, particle_mass)
    meshes = sim.make_meshes(collider_objects)
    wp.synchronize()
    return WarpCheckStats(
        n_strands=n_strands,
        points_per_strand=pps,
        n_points=int(len(world)),
        n_segments=int(n_strands * (pps - 1)),
        root_locked_points=locked,
        n_vertices=meshes.n_vertices,
        n_triangles=meshes.n_triangles,
        device=sim.device,
        device_name=sim.device_name,
        device_arch=sim.device_arch,
        warp_version=str(getattr(wp, "__version__", "?")),
    )


def simulate(
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
    stretch_compliance: float,
    bend_compliance: float,
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
) -> WarpSimStats:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if not collider_objects:
        raise ValueError("expected at least one Mesh collider")
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame < start_frame:
        raise ValueError("End Frame must be >= Start Frame")
    bake_mode = str(bake_mode).strip().upper()
    if bake_mode not in {"CACHE", "FINAL", "KEYFRAMES"}:
        bake_mode = "CACHE"
    guide_decimation = max(1, int(guide_decimation))

    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    frames = list(range(start_frame, end_frame + 1))
    t0 = time.perf_counter()
    max_substeps_seen = 1
    frame_steps = 0
    total_substeps = 0
    max_auto_move_mm = 0.0
    total_hits = 0
    post_keep_hits_total = 0
    post_keep_active_strands_total = 0
    body_hard_guard_points_total = 0
    body_fk_repair_strands_total = 0
    body_fk_repair_points_total = 0
    body_fk_escape_points_total = 0
    body_fk_failed_points_total = 0
    body_fk_velocity_zeroed_total = 0
    last_triangles = 0
    max_keep_length_error_mm = 0.0
    success = False
    cache_path = ""
    target_worlds = {}
    offsets = {}
    global _CACHE_MUTED
    previous_cache_muted = _CACHE_MUTED
    _CACHE_MUTED = True

    try:
        clear_runtime_cache(curves_obj)
        scene.frame_set(start_frame)
        pps, n_strands = _uniform_points_per_strand(curves_obj)
        locked = max(1, min(int(root_locked_points), pps))
        keep_length_source_frame = 1
        target_read_frames = sorted({int(frame) for frame in frames + [original_frame, keep_length_source_frame]})
        target_worlds, offsets = _read_targets_for_frames(curves_obj, target_read_frames)
        init_eval = target_worlds[start_frame]
        keep_rest_lengths = None
        keep_fallback_dirs = None
        if keep_length:
            keep_rest_lengths, keep_fallback_dirs = _segment_lengths_and_dirs(
                target_worlds[keep_length_source_frame],
                pps,
            )
        guide_indices = _decimated_guide_indices(n_strands, guide_decimation)
        guide_point_indices = _guide_point_indices(guide_indices, pps)
        post_keep_contact_ttl = np.zeros(len(guide_indices), dtype=np.int32)
        body_fk_contact_ttl = np.zeros(n_strands, dtype=np.int32)
        body_hard_guard_obj = collider_objects[0]
        body_hard_guard_seed = _head_seed_world(body_hard_guard_obj)
        guide_keep_rest_lengths = None
        if keep_rest_lengths is not None:
            guide_keep_rest_lengths = np.ascontiguousarray(keep_rest_lengths[guide_indices], dtype=np.float32)
        sim_init = np.ascontiguousarray(init_eval[guide_point_indices], dtype=np.float32)
        guide_nearest = None
        guide_weights = None
        if len(guide_indices) != n_strands:
            guide_nearest, guide_weights = _guide_interpolation_weights(
                init_eval,
                pps,
                guide_indices,
            )
        simulator = WarpJointSimulator(sim_init, pps, locked, particle_mass)
        if guide_keep_rest_lengths is not None:
            simulator.set_keep_rest_lengths(guide_keep_rest_lengths)
        prev_targets = sim_init.copy()
        current_sim_world = sim_init.copy()
        current_world = init_eval.copy()
        if keep_rest_lengths is not None and keep_fallback_dirs is not None:
            current_world, keep_error = _keep_length_fk(
                current_world,
                pps,
                keep_rest_lengths,
                keep_fallback_dirs,
            )
            max_keep_length_error_mm = max(max_keep_length_error_mm, keep_error)
            current_sim_world = np.ascontiguousarray(current_world[guide_point_indices], dtype=np.float32)
            simulator.set_positions(current_sim_world)
            prev_targets = current_sim_world.copy()
        current_world, guard_fixed, guard_strands = _apply_body_seed_hard_guard(
            current_world,
            body_hard_guard_obj,
            float(collision_margin_m),
            body_hard_guard_seed,
            points_per_strand=pps,
        )
        body_hard_guard_points_total += guard_fixed
        body_zero_strands = guard_strands.astype(np.int32, copy=True)
        if keep_rest_lengths is not None and keep_fallback_dirs is not None:
            initial_fk_active = np.ones(n_strands, dtype=np.int32)
            current_world, fk_stats, fk_strands = _apply_body_fk_hard_repair(
                current_world,
                pps,
                keep_rest_lengths,
                keep_fallback_dirs,
                body_hard_guard_obj,
                float(collision_margin_m),
                body_hard_guard_seed,
                root_locked_points=locked,
                active_strands=initial_fk_active,
            )
            if fk_stats.points:
                body_fk_repair_strands_total += fk_stats.strands
                body_fk_repair_points_total += fk_stats.points
                body_fk_escape_points_total += fk_stats.escape_points
                body_fk_failed_points_total += fk_stats.failed_points
                body_fk_contact_ttl[fk_strands != 0] = BODY_FK_REPAIR_TTL_FRAMES
                body_zero_strands |= fk_strands.astype(np.int32, copy=False)
        if np.any(body_zero_strands):
            guide_zero = body_zero_strands[guide_indices].astype(np.int32, copy=False)
            body_fk_velocity_zeroed_total += simulator.zero_strand_velocities(guide_zero)
        if guard_fixed:
            current_sim_world = np.ascontiguousarray(current_world[guide_point_indices], dtype=np.float32)
            simulator.set_positions(current_sim_world)
            prev_targets = current_sim_world.copy()
        elif keep_rest_lengths is not None and keep_fallback_dirs is not None and body_fk_repair_points_total:
            current_sim_world = np.ascontiguousarray(current_world[guide_point_indices], dtype=np.float32)
            simulator.set_positions(current_sim_world)
            prev_targets = current_sim_world.copy()
        baked = {start_frame: current_world.copy()}
        show_error = _show_sim_frame(
            curves_obj,
            start_frame,
            current_world,
            offset=offsets[start_frame],
            pps=pps,
            keep_rest_lengths=keep_rest_lengths,
            keep_fallback_dirs=keep_fallback_dirs,
            body_hard_guard_obj=body_hard_guard_obj,
            body_hard_guard_margin=float(collision_margin_m),
            body_hard_guard_seed=body_hard_guard_seed,
            body_fk_root_locked_points=locked,
        )
        max_keep_length_error_mm = max(max_keep_length_error_mm, show_error)

        dt_frame = float(scene.render.fps_base) / max(float(scene.render.fps), 1.0)
        for frame in frames[1:]:
            scene.frame_set(frame)
            eval_world = target_worlds[frame]
            eval_sim_world = np.ascontiguousarray(eval_world[guide_point_indices], dtype=np.float32)
            target_move_mm = _target_motion_mm(prev_targets, eval_sim_world, simulator.inv_mass_np)
            inertial_move_mm = simulator.estimated_free_move_mm(dt_frame, gravity)
            move_mm = max(target_move_mm, inertial_move_mm)
            max_auto_move_mm = max(max_auto_move_mm, move_mm)
            max_move = max(float(max_move_per_substep_m), 1.0e-6)
            substeps = max(1, int(math.ceil((move_mm * 1.0e-3) / max_move)))
            substeps = min(max(1, int(max_substeps)), substeps)
            max_substeps_seen = max(max_substeps_seen, substeps)
            frame_steps += 1
            total_substeps += substeps
            print(
                "Yurameki Warp "
                f"frame={frame}/{end_frame} "
                f"substeps={substeps} "
                f"auto_move_mm={move_mm:.3f} "
                f"guides={len(guide_indices)}/{n_strands} "
                f"device={simulator.device} "
                f"arch=sm_{simulator.device_arch}"
            )
            current_sim_world, hits, last_triangles, contact_strands = simulator.simulate_frame(
                prev_targets,
                eval_sim_world,
                collider_objects,
                substeps,
                dt_frame,
                gravity,
                damping,
                float(max_velocity_mps),
                int(iterations),
                float(stretch_compliance),
                float(bend_compliance),
                float(collision_margin_m),
                float(collision_search_m),
                float(collision_max_correction_m),
                float(collision_response),
                float(collision_velocity_damping),
                int(collision_passes),
                int(post_collision_iterations),
            )
            post_hits = 0
            post_contacts = np.zeros_like(contact_strands)
            if guide_nearest is None or guide_weights is None:
                current_world = current_sim_world.copy()
            else:
                current_world = _restore_decimated_strands(
                    eval_world,
                    eval_sim_world,
                    current_sim_world,
                    pps,
                    guide_nearest,
                    guide_weights,
                )
            if keep_rest_lengths is not None and keep_fallback_dirs is not None:
                current_world, keep_error = _keep_length_fk(
                    current_world,
                    pps,
                    keep_rest_lengths,
                    keep_fallback_dirs,
                )
                max_keep_length_error_mm = max(max_keep_length_error_mm, keep_error)
                current_sim_world = np.ascontiguousarray(current_world[guide_point_indices], dtype=np.float32)
                simulator.set_positions(current_sim_world)
                active_strands = (contact_strands != 0) | (post_keep_contact_ttl > 0)
                if guide_keep_rest_lengths is not None and np.any(active_strands):
                    post_keep_active_strands_total += int(np.count_nonzero(active_strands))
                    current_sim_world, post_hits, post_contacts = simulator.post_keep_length_collision(
                        active_strands.astype(np.int32, copy=False),
                        float(collision_margin_m),
                        float(collision_search_m),
                        float(collision_max_correction_m),
                        float(collision_response),
                        max(1, int(post_collision_iterations)),
                    )
                    post_keep_hits_total += post_hits
                    if guide_nearest is None or guide_weights is None:
                        current_world = current_sim_world.copy()
                    else:
                        current_world = _restore_decimated_strands(
                            eval_world,
                            eval_sim_world,
                            current_sim_world,
                            pps,
                            guide_nearest,
                            guide_weights,
                        )
                        current_world, keep_error = _keep_length_fk(
                            current_world,
                            pps,
                            keep_rest_lengths,
                            keep_fallback_dirs,
                        )
                        max_keep_length_error_mm = max(max_keep_length_error_mm, keep_error)
                        current_sim_world = np.ascontiguousarray(current_world[guide_point_indices], dtype=np.float32)
                        simulator.set_positions(current_sim_world)
                keep_contacts = (contact_strands != 0) | (post_contacts != 0)
                post_keep_contact_ttl = np.maximum(post_keep_contact_ttl - 1, 0)
                post_keep_contact_ttl[keep_contacts] = POST_KEEP_CONTACT_TTL_FRAMES
                guide_active_for_fk = keep_contacts | (post_keep_contact_ttl > 0)
            else:
                guide_active_for_fk = contact_strands != 0
            current_world, guard_fixed, guard_strands = _apply_body_seed_hard_guard(
                current_world,
                body_hard_guard_obj,
                float(collision_margin_m),
                body_hard_guard_seed,
                points_per_strand=pps,
            )
            body_hard_guard_points_total += guard_fixed
            body_fk_points_this_frame = 0
            body_zero_strands = guard_strands.astype(np.int32, copy=True)
            if keep_rest_lengths is not None and keep_fallback_dirs is not None:
                body_fk_active = _full_active_strands_from_guides(
                    n_strands,
                    guide_indices,
                    guide_active_for_fk,
                    guide_nearest,
                )
                body_fk_active |= guard_strands.astype(np.int32, copy=False)
                body_fk_active |= (body_fk_contact_ttl > 0).astype(np.int32, copy=False)
                body_fk_contact_ttl = np.maximum(body_fk_contact_ttl - 1, 0)
                current_world, fk_stats, fk_strands = _apply_body_fk_hard_repair(
                    current_world,
                    pps,
                    keep_rest_lengths,
                    keep_fallback_dirs,
                    body_hard_guard_obj,
                    float(collision_margin_m),
                    body_hard_guard_seed,
                    root_locked_points=locked,
                    active_strands=body_fk_active,
                )
                body_fk_points_this_frame = fk_stats.points
                if fk_stats.points:
                    body_fk_repair_strands_total += fk_stats.strands
                    body_fk_repair_points_total += fk_stats.points
                    body_fk_escape_points_total += fk_stats.escape_points
                    body_fk_failed_points_total += fk_stats.failed_points
                    body_fk_contact_ttl[fk_strands != 0] = BODY_FK_REPAIR_TTL_FRAMES
                    body_zero_strands |= fk_strands.astype(np.int32, copy=False)
            if np.any(body_zero_strands):
                guide_zero = body_zero_strands[guide_indices].astype(np.int32, copy=False)
                body_fk_velocity_zeroed_total += simulator.zero_strand_velocities(guide_zero)
            if guard_fixed or body_fk_points_this_frame:
                current_sim_world = np.ascontiguousarray(current_world[guide_point_indices], dtype=np.float32)
                simulator.set_positions(current_sim_world)
            total_hits += hits + post_hits
            prev_targets = current_sim_world.copy() if (keep_length or guard_fixed or body_fk_points_this_frame) else eval_sim_world.copy()
            baked[frame] = current_world.copy()
            show_error = _show_sim_frame(
                curves_obj,
                frame,
                current_world,
                offset=offsets[frame],
                pps=pps,
                keep_rest_lengths=keep_rest_lengths,
                keep_fallback_dirs=keep_fallback_dirs,
                body_hard_guard_obj=body_hard_guard_obj,
                body_hard_guard_margin=float(collision_margin_m),
                body_hard_guard_seed=body_hard_guard_seed,
                body_fk_root_locked_points=locked,
            )
            max_keep_length_error_mm = max(max_keep_length_error_mm, show_error)

        if bake_mode == "KEYFRAMES":
            _bake_position_keyframes(
                curves_obj,
                frames,
                baked,
                offsets,
                pps=pps,
                keep_rest_lengths=keep_rest_lengths,
                keep_fallback_dirs=keep_fallback_dirs,
                body_hard_guard_obj=body_hard_guard_obj,
                body_hard_guard_margin=float(collision_margin_m),
                body_hard_guard_seed=body_hard_guard_seed,
                body_fk_root_locked_points=locked,
            )
        elif bake_mode == "CACHE":
            cache = _register_sim_cache(
                curves_obj,
                frames,
                baked,
                offsets,
                pps=pps,
                keep_rest_lengths=keep_rest_lengths,
                keep_fallback_dirs=keep_fallback_dirs,
                body_hard_guard_obj=body_hard_guard_obj,
                body_hard_guard_margin=float(collision_margin_m),
                body_hard_guard_seed=body_hard_guard_seed,
                body_fk_root_locked_points=locked,
            )
            cache_path = cache.path
        else:
            show_error = _show_sim_frame(
                curves_obj,
                end_frame,
                baked[end_frame],
                offset=offsets[end_frame],
                pps=pps,
                keep_rest_lengths=keep_rest_lengths,
                keep_fallback_dirs=keep_fallback_dirs,
                body_hard_guard_obj=body_hard_guard_obj,
                body_hard_guard_margin=float(collision_margin_m),
                body_hard_guard_seed=body_hard_guard_seed,
                body_fk_root_locked_points=locked,
            )
            max_keep_length_error_mm = max(max_keep_length_error_mm, show_error)
        success = True
    finally:
        _CACHE_MUTED = previous_cache_muted
        if not success:
            scene.frame_set(original_frame)
            if original_frame in target_worlds and original_frame in offsets:
                _write_world_points(
                    curves_obj,
                    target_worlds[original_frame],
                    offset=offsets[original_frame],
                )
                _force_viewport_refresh()

    return WarpSimStats(
        start_frame=start_frame,
        end_frame=end_frame,
        n_frames=len(frames),
        n_strands=n_strands,
        simulated_strands=int(len(guide_indices)),
        guide_decimation=guide_decimation,
        points_per_strand=pps,
        root_locked_points=locked,
        frame_steps=frame_steps,
        total_substeps=total_substeps,
        max_substeps=max_substeps_seen,
        max_auto_move_mm=max_auto_move_mm,
        total_hits=total_hits,
        n_triangles_last=last_triangles,
        bake_mode=bake_mode,
        max_velocity_mps=float(max_velocity_mps),
        collision_max_correction_mm=float(collision_max_correction_m) * 1000.0,
        collision_response=min(max(float(collision_response), 0.0), 1.0),
        collision_velocity_damping=min(max(float(collision_velocity_damping), 0.0), 1.0),
        post_keep_collision_hits=post_keep_hits_total,
        post_keep_active_strands=post_keep_active_strands_total,
        body_hard_guard_points=body_hard_guard_points_total,
        body_fk_repair_strands=body_fk_repair_strands_total,
        body_fk_repair_points=body_fk_repair_points_total,
        body_fk_escape_points=body_fk_escape_points_total,
        body_fk_failed_points=body_fk_failed_points_total,
        body_fk_velocity_zeroed=body_fk_velocity_zeroed_total,
        keep_length=bool(keep_length),
        keep_length_source_frame=1,
        max_keep_length_error_mm=max_keep_length_error_mm,
        cache_path=cache_path,
        device=simulator.device,
        device_name=simulator.device_name,
        device_arch=simulator.device_arch,
        elapsed_sec=time.perf_counter() - t0,
    )
