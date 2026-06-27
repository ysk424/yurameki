"""CUDA body collision using NVIDIA Warp's triangle-mesh queries.

The collision kernels are unchanged from Tokoya. The class can be fed body
geometry two ways: from Blender (``body_name`` -> evaluated mesh, lazy ``bpy``
import) or headless (``triangles=(vertices, indices)`` in world space). For an
animated body, call ``update_mesh(vertices)`` once per frame.
"""
from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _point_collision(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    points_per_strand: int,
    margin: float,
    search_distance: float,
    allow_sweep: int,
):
    i = wp.tid()
    point_index = i % points_per_strand
    if point_index < 2:
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
        ray = wp.mesh_query_ray(
            mesh, old_position, direction, distance
        )
        if ray.result and wp.dot(delta, ray.normal) < 0.0:
            new_position = (
                old_position + direction * ray.t + ray.normal * margin
            )
            normal = ray.normal
            contacted = 1

    if contacted == 0:
        query = wp.mesh_query_point_no_sign(
            mesh, new_position, search_distance
        )
        if query.result:
            closest = wp.mesh_eval_position(
                mesh, query.face, query.u, query.v
            )
            face_normal = wp.mesh_eval_face_normal(mesh, query.face)
            signed_distance = wp.dot(new_position - closest, face_normal)
            if signed_distance < margin:
                new_position = closest + face_normal * margin
                normal = face_normal
                contacted = 1

    if contacted == 1:
        velocity = velocities[i]
        normal_speed = wp.dot(velocity, normal)
        if normal_speed < 0.0:
            velocities[i] = velocity - normal * normal_speed
        predicted[i] = new_position


@wp.kernel
def _segment_collision(
    mesh: wp.uint64,
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    points_per_strand: int,
    margin: float,
    parity: int,
    final_cleanup: int,
):
    segment_id = wp.tid()
    segments_per_strand = points_per_strand - 1
    strand = segment_id // segments_per_strand
    segment = segment_id % segments_per_strand
    if segment % 2 != parity:
        return

    i = strand * points_per_strand + segment
    j = i + 1
    p0 = predicted[i]
    p1 = predicted[j]
    delta = p1 - p0
    distance = wp.length(delta)
    if distance < 1.0e-9:
        return

    direction = delta / distance
    ray = wp.mesh_query_ray(mesh, p0, direction, distance)
    if not ray.result or ray.t <= 1.0e-6 or ray.t >= distance - 1.0e-6:
        return

    target = p0 + direction * ray.t + ray.normal * margin
    if final_cleanup == 1:
        if (j % points_per_strand) >= 2:
            predicted[j] = target
            velocity = velocities[j]
            normal_speed = wp.dot(velocity, ray.normal)
            if normal_speed < 0.0:
                velocities[j] = velocity - ray.normal * normal_speed
        return

    fraction = ray.t / distance
    wi = 0.0
    wj = 0.0
    if (i % points_per_strand) >= 2:
        wi = 1.0
    if (j % points_per_strand) >= 2:
        wj = 1.0
    denom = wi * (1.0 - fraction) * (1.0 - fraction)
    denom += wj * fraction * fraction
    if denom <= 1.0e-12:
        return

    correction = ray.normal * margin
    if wi > 0.0:
        predicted[i] = p0 + correction * (
            (1.0 - fraction) * wi / denom
        )
        velocity = velocities[i]
        normal_speed = wp.dot(velocity, ray.normal)
        if normal_speed < 0.0:
            velocities[i] = velocity - ray.normal * normal_speed
    if wj > 0.0:
        predicted[j] = p1 + correction * (fraction * wj / denom)
        velocity = velocities[j]
        normal_speed = wp.dot(velocity, ray.normal)
        if normal_speed < 0.0:
            velocities[j] = velocity - ray.normal * normal_speed


def _evaluated_mesh_arrays(mesh_name: str):
    import bpy
    body = bpy.data.objects.get(mesh_name)
    if body is None or body.type != "MESH":
        raise ValueError(f"Collision mesh {mesh_name!r} not found")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = body.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertex_count = len(mesh.vertices)
        triangle_count = len(mesh.loop_triangles)
        vertices = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", vertices)
        vertices = vertices.reshape(-1, 3)
        matrix = np.array(evaluated.matrix_world, dtype=np.float32)
        homogeneous = np.column_stack(
            (vertices, np.ones(vertex_count, dtype=np.float32))
        )
        vertices = (homogeneous @ matrix.T)[:, :3].astype(
            np.float32, copy=False
        )
        indices = np.empty(triangle_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", indices)
        return vertices, indices
    finally:
        evaluated.to_mesh_clear()


def _evaluated_collision_arrays(mesh_names):
    names = [name for name in mesh_names if name]
    if not names:
        raise ValueError("No collision meshes selected")
    vertex_parts = []
    index_parts = []
    vertex_offset = 0
    for name in names:
        vertices, indices = _evaluated_mesh_arrays(name)
        vertex_parts.append(vertices)
        index_parts.append(indices.reshape(-1) + vertex_offset)
        vertex_offset += len(vertices)
    return (
        np.ascontiguousarray(np.concatenate(vertex_parts, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(index_parts, axis=0), dtype=np.int32),
    )


def _evaluated_body_arrays(body_name: str):
    return _evaluated_collision_arrays([body_name])


class WarpBodyCollider:
    def __init__(
        self,
        n_total: int,
        points_per_strand: int,
        margin: float,
        search_distance: float,
        body_name: str = None,
        collider_names=None,
        triangles=None,
    ):
        if not wp.is_cuda_available():
            raise RuntimeError("NVIDIA Warp CUDA device is unavailable")
        self.device = "cuda:0"
        self.n_total = n_total
        self.points_per_strand = points_per_strand
        self.n_segments = (
            n_total // points_per_strand * (points_per_strand - 1)
        )
        self.margin = float(margin)
        self.search_distance = float(search_distance)

        if triangles is not None:
            vertices, indices = triangles
            vertices = np.ascontiguousarray(vertices, dtype=np.float32)
            indices = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1)
        elif collider_names is not None:
            vertices, indices = _evaluated_collision_arrays(collider_names)
        else:
            vertices, indices = _evaluated_body_arrays(body_name)
        self.points = wp.array(vertices, dtype=wp.vec3, device=self.device)
        self.mesh = wp.Mesh(
            points=self.points,
            indices=wp.array(
                indices, dtype=wp.int32, device=self.device
            ),
        )
        self.positions = wp.empty(
            n_total, dtype=wp.vec3, device=self.device
        )
        self.predicted = wp.empty(
            n_total, dtype=wp.vec3, device=self.device
        )
        self.velocities = wp.empty(
            n_total, dtype=wp.vec3, device=self.device
        )

    def update_mesh(self, vertices):
        """Refresh body vertex positions in world space (topology fixed).

        Call once per frame for an animated body. Indices/topology are
        unchanged; only the BVH is refit.
        """
        self.points.assign(np.ascontiguousarray(vertices, dtype=np.float32))
        self.mesh.refit()

    def __call__(
        self,
        pos_np,
        pred_np,
        vel_np,
        allow_sweep=True,
        final_cleanup=False,
    ):
        self.positions.assign(
            np.ascontiguousarray(pos_np, dtype=np.float32)
        )
        self.predicted.assign(
            np.ascontiguousarray(pred_np, dtype=np.float32)
        )
        self.velocities.assign(
            np.ascontiguousarray(vel_np, dtype=np.float32)
        )
        self.apply_device(
            self.positions,
            self.predicted,
            self.velocities,
            allow_sweep=allow_sweep,
            final_cleanup=final_cleanup,
        )
        wp.synchronize()
        pred_np[:] = self.predicted.numpy()
        vel_np[:] = self.velocities.numpy()

    def apply_device(
        self,
        positions,
        predicted,
        velocities,
        allow_sweep=True,
        final_cleanup=False,
    ):
        """Run collision in-place on Warp CUDA arrays without host copies."""
        wp.launch(
            _point_collision,
            dim=self.n_total,
            inputs=[
                self.mesh.id,
                positions,
                predicted,
                velocities,
                self.points_per_strand,
                self.margin,
                self.search_distance,
                int(bool(allow_sweep)),
            ],
            device=self.device,
        )
        cleanup_passes = 4 if final_cleanup else 1
        for _ in range(cleanup_passes):
            for parity in (0, 1):
                wp.launch(
                    _segment_collision,
                    dim=self.n_segments,
                    inputs=[
                        self.mesh.id,
                        predicted,
                        velocities,
                        self.points_per_strand,
                        self.margin,
                        parity,
                        int(bool(final_cleanup)),
                    ],
                    device=self.device,
                )
