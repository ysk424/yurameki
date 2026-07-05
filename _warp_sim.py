"""NVIDIA Warp joint-chain simulation for Yurameki.

This path intentionally uses the existing Blender Curves joints. It does not
resample into cylinders. The first N points of every strand are kinematic
constraints following the evaluated Curves shape; the remaining points are
simulated with distance, bend, gravity, damping, and Warp Mesh collision.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import bpy
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
    device: str
    device_name: str
    device_arch: int
    elapsed_sec: float


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
    margin: float,
    search_distance: float,
    max_correction: float,
    collision_response: float,
    allow_sweep: int,
    contact_mask: wp.array(dtype=wp.int32),
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
    margin: float,
    search_distance: float,
    max_correction: float,
    collision_response: float,
    allow_sweep: int,
    contact_mask: wp.array(dtype=wp.int32),
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
        velocity = velocities[j]
        normal_speed = wp.dot(velocity, normal)
        if normal_speed < 0.0:
            velocities[j] = velocity - normal * normal_speed
        wp.atomic_add(hit_count, 0, 1)


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


def _rest_lengths(points: np.ndarray, pps: int) -> tuple[np.ndarray, np.ndarray]:
    strands = points.reshape(-1, pps, 3)
    seg = np.linalg.norm(strands[:, 1:, :] - strands[:, :-1, :], axis=2)
    bend = np.linalg.norm(strands[:, 2:, :] - strands[:, :-2, :], axis=2)
    return (
        np.ascontiguousarray(np.maximum(seg, 1.0e-6).reshape(-1), dtype=np.float32),
        np.ascontiguousarray(np.maximum(bend, 1.0e-6).reshape(-1), dtype=np.float32),
    )


def _inverse_mass(n_strands: int, pps: int, root_locked_points: int) -> np.ndarray:
    locked = max(1, min(int(root_locked_points), pps))
    inv = np.ones(n_strands * pps, dtype=np.float32)
    for strand in range(n_strands):
        base = strand * pps
        inv[base:base + locked] = 0.0
    return inv


def _target_motion_mm(prev_targets: np.ndarray, next_targets: np.ndarray,
                      inv_mass: np.ndarray) -> float:
    locked = inv_mass <= 0.0
    if not np.any(locked):
        return 0.0
    delta = next_targets[locked] - prev_targets[locked]
    return float(np.max(np.linalg.norm(delta, axis=1)) * 1000.0) if len(delta) else 0.0


def _bake_position_keyframes(curves_obj, frames: list[int],
                             baked: dict[int, np.ndarray],
                             offsets: dict[int, np.ndarray]) -> None:
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    n_total = len(attr.data)
    if n_total == 0:
        raise ValueError("Curves has no points")

    scene = bpy.context.scene
    frame_numbers = np.asarray(frames, dtype=np.float32)
    local_values = np.empty((len(frames), n_total, 3), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        scene.frame_set(frame)
        local_values[frame_index] = _world_to_local_points(
            curves_obj,
            baked[frame],
            offset=offsets[frame],
        )

    data = curves_obj.data
    anim = data.animation_data_create()
    if anim.action is None:
        anim.action = bpy.data.actions.new(f"Yurameki Bake {curves_obj.name}")
    action = anim.action

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
        self.hit_count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def set_targets(self, start: np.ndarray, end: np.ndarray) -> None:
        self.target_start.assign(np.ascontiguousarray(start, dtype=np.float32))
        self.target_end.assign(np.ascontiguousarray(end, dtype=np.float32))

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
                    float(margin),
                    float(search_distance),
                    max_correction,
                    collision_response,
                    int(bool(allow_sweep)),
                    self.contact_mask,
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
                    float(margin),
                    float(search_distance),
                    max_correction,
                    collision_response,
                    int(bool(allow_sweep)),
                    self.contact_mask,
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
                            self.hit_count,
                        ],
                        device=self.device,
                    )

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
    ) -> tuple[np.ndarray, int, int]:
        meshes = self.make_meshes(collider_objects)
        self.set_targets(target_start, target_end)
        self.hit_count.assign(np.zeros(1, dtype=np.int32))
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
        return self.pos.numpy().astype(np.float32, copy=True), hits, meshes.n_triangles


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
    if bake_mode not in {"FINAL", "KEYFRAMES"}:
        bake_mode = "KEYFRAMES"

    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    frames = list(range(start_frame, end_frame + 1))
    t0 = time.perf_counter()
    max_substeps_seen = 1
    frame_steps = 0
    total_substeps = 0
    max_auto_move_mm = 0.0
    total_hits = 0
    last_triangles = 0
    success = False

    try:
        scene.frame_set(start_frame)
        pps, n_strands = _uniform_points_per_strand(curves_obj)
        locked = max(1, min(int(root_locked_points), pps))
        init_eval, init_original = _read_world(curves_obj)
        offset = init_eval - init_original
        simulator = WarpJointSimulator(init_eval, pps, locked, particle_mass)
        prev_targets = init_eval.copy()
        current_world = init_eval.copy()
        offsets = {start_frame: offset}
        baked = {start_frame: current_world.copy()}

        dt_frame = float(scene.render.fps_base) / max(float(scene.render.fps), 1.0)
        for frame in frames[1:]:
            scene.frame_set(frame)
            eval_world, original_world = _read_world(curves_obj)
            offsets[frame] = eval_world - original_world
            target_move_mm = _target_motion_mm(prev_targets, eval_world, simulator.inv_mass_np)
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
                f"device={simulator.device} "
                f"arch=sm_{simulator.device_arch}"
            )
            current_world, hits, last_triangles = simulator.simulate_frame(
                prev_targets,
                eval_world,
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
            total_hits += hits
            prev_targets = eval_world.copy()
            baked[frame] = current_world.copy()

        if bake_mode == "KEYFRAMES":
            _bake_position_keyframes(curves_obj, frames, baked, offsets)
        else:
            scene.frame_set(end_frame)
            _write_world_points(curves_obj, baked[end_frame], offset=offsets[end_frame])
        success = True
    finally:
        if not success:
            scene.frame_set(original_frame)

    return WarpSimStats(
        start_frame=start_frame,
        end_frame=end_frame,
        n_frames=len(frames),
        n_strands=n_strands,
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
        device=simulator.device,
        device_name=simulator.device_name,
        device_arch=simulator.device_arch,
        elapsed_sec=time.perf_counter() - t0,
    )
