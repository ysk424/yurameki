"""Frame-range gravity bake for Yurameki 0.5.x.

This pass starts from the current groomed Curves state, follows evaluated root
motion per frame, applies gravity with FK length preservation, resolves the
body collider on CUDA, stores all frame results in memory, then bakes keyframes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import bpy
import numpy as np

from . import cuda_collider
from . import solver_interface as si


@dataclass
class GravityBakeStats:
    start_frame: int
    end_frame: int
    n_frames: int
    n_strands: int
    points_per_strand: int
    max_substeps: int
    max_root_move_mm: float
    elapsed_sec: float


def _curve_spans(curves_obj):
    return [
        (int(curve.first_point_index), int(curve.points_length))
        for curve in curves_obj.data.curves
    ]


def _points_per_strand(curves_obj, require_uniform: bool = True) -> tuple[int, int]:
    spans = _curve_spans(curves_obj)
    if not spans:
        raise ValueError("Curves object has no strands")
    lengths = sorted({length for _start, length in spans})
    if min(lengths) < 2:
        raise ValueError(f"{min(lengths)} points per strand is too small")
    if require_uniform and len(lengths) != 1:
        raise ValueError(f"strands have different point counts: {lengths[:8]}")
    return int(max(lengths)), len(spans)


def _write_and_keyframe(curves_obj, world_pts: np.ndarray, frame: int) -> None:
    si._write_world_points(curves_obj, world_pts)
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    # Curves point positions are individual RNA vectors; keyframing each point is
    # heavy, but this bake happens once after simulation so solver speed stays OK.
    for item in attr.data:
        item.keyframe_insert("vector", frame=frame)


def prepare_gravity_sim(curves_obj, collider_obj, points_per_strand: int,
                        radius_m: float, sort_axis: str = "Z",
                        cylinder_length_m: float = si.DEFAULT_CYLINDER_LENGTH
                        ) -> dict:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if collider_obj is None or collider_obj.type != "MESH":
        raise ValueError("expected one Mesh collider")
    pps, strands = _points_per_strand(curves_obj, require_uniform=False)
    if pps != int(points_per_strand):
        raise ValueError(f"points_per_strand mismatch: UI={points_per_strand}, data={pps}")
    result = cuda_collider.detect_capsule_mesh(
        curves_obj,
        collider_obj,
        points_per_strand=points_per_strand,
        radius_m=radius_m,
        sort_axis=sort_axis,
        cylinder_length_m=cylinder_length_m,
    )
    return {
        "n_strands": strands,
        "points_per_strand": pps,
        "n_cylinders": result.n_cylinders,
        "n_triangles": result.n_triangles,
        "hit_count": result.hit_count,
    }


def simulate_gravity_bake(curves_obj, collider_obj, points_per_strand: int,
                          start_frame: int, end_frame: int,
                          gravity_step_m: float = 0.001,
                          radius_m: float = 0.0005,
                          collider_substeps: int = 4,
                          collider_max_move_m: float = 0.001) -> GravityBakeStats:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if collider_obj is None or collider_obj.type != "MESH":
        raise ValueError("expected one Mesh collider")
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame < start_frame:
        raise ValueError("End Frame must be >= Start Frame")

    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    t0 = time.perf_counter()
    try:
        scene.frame_set(start_frame)
        pps, n_strands = _points_per_strand(curves_obj)
        if pps != int(points_per_strand):
            raise ValueError(f"points_per_strand mismatch: UI={points_per_strand}, data={pps}")

        eval_world, _original_world = si._read_world(curves_obj)
        sim = eval_world.reshape(n_strands, pps, 3).astype(np.float32, copy=True)
        seg_lens = np.linalg.norm(sim[:, 1:, :] - sim[:, :-1, :], axis=2).astype(np.float32)
        roots_prev = sim[:, 0, :].copy()
        last_z_strand = int(np.lexsort((np.arange(n_strands), roots_prev[:, 2]))[-1])

        frames = list(range(start_frame, end_frame + 1))
        baked = {}
        max_substeps = 1
        max_root_move = 0.0
        for frame in frames:
            scene.frame_set(frame)
            eval_world, _unused_original = si._read_world(curves_obj)
            eval_roots = eval_world.reshape(n_strands, pps, 3)[:, 0, :].astype(np.float32, copy=True)
            vertices, triangles = cuda_collider.mesh_world_triangles(collider_obj)

            if frame == start_frame:
                baked[frame] = sim.reshape(n_strands * pps, 3).astype(np.float32, copy=True)
                roots_prev = eval_roots
                continue

            move_m = float(np.linalg.norm(eval_roots[last_z_strand] - roots_prev[last_z_strand]))
            substeps = max(1, int(math.ceil(move_m * 1000.0)))
            max_substeps = max(max_substeps, substeps)
            max_root_move = max(max_root_move, move_m)

            sim, _hit_count = cuda_collider.simulate_gravity_frame_arrays(
                sim,
                roots_prev,
                eval_roots,
                seg_lens,
                vertices,
                triangles,
                gravity_step_m=float(gravity_step_m),
                radius_m=radius_m,
                n_substeps=substeps,
                collider_substeps=collider_substeps,
                max_move_m=collider_max_move_m,
            )

            roots_prev = eval_roots
            baked[frame] = sim.reshape(n_strands * pps, 3).astype(np.float32, copy=True)

        for frame in frames:
            _write_and_keyframe(curves_obj, baked[frame], frame)

    finally:
        scene.frame_set(original_frame)

    return GravityBakeStats(
        start_frame=start_frame,
        end_frame=end_frame,
        n_frames=end_frame - start_frame + 1,
        n_strands=n_strands,
        points_per_strand=pps,
        max_substeps=max_substeps,
        max_root_move_mm=max_root_move * 1000.0,
        elapsed_sec=time.perf_counter() - t0,
    )
