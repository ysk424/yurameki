"""Frame-range chain simulation bake for Yurameki 0.6.x.

This pass starts from the current groomed Curves state, subdivides each strand
into a fixed-length simulation chain, follows evaluated root motion per frame,
propagates root motion along the chain with linear attenuation, resolves the
body collider on CUDA, stores frame results in memory, then bakes keyframes.
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
    total_hits: int
    interpolation_steps: int
    sim_points_per_strand: int
    target_segment_cm: float
    max_segment_mm: float
    propagation_length_cm: float
    memory_height_m: float
    memory_strength: float
    memory_points: int
    bake_mode: str
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


def _write_and_keyframe(curves_obj, world_pts: np.ndarray, frame: int, offset=None) -> None:
    si._write_world_points(curves_obj, world_pts, offset=offset)
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    # Attribute values are not ID datablocks in Blender 5.x, so keyframe from
    # the Curves datablock with an explicit RNA path.
    for i in range(len(attr.data)):
        curves_obj.data.keyframe_insert(
            f'attributes["position"].data[{i}].vector',
            frame=frame,
        )


def _segment_divisions(points: np.ndarray,
                       target_segment_length_m: float,
                       interpolation_steps: int) -> np.ndarray:
    """Choose a uniform per-source-segment division count for all strands."""
    target = max(1.0e-5, float(target_segment_length_m))
    interpolation_steps = max(0, int(interpolation_steps))
    min_divisions = interpolation_steps + 1
    seg_len = np.linalg.norm(points[:, 1:, :] - points[:, :-1, :], axis=2)
    max_seg_len = np.max(seg_len, axis=0)
    divisions = np.ceil(max_seg_len / target).astype(np.int32)
    return np.maximum(divisions, min_divisions)


def _resample_chain_with_divisions(points: np.ndarray,
                                   divisions: np.ndarray) -> np.ndarray:
    """Insert simulation points inside every source segment."""
    n_strands, source_pps, _xyz = points.shape
    if len(divisions) != source_pps - 1:
        raise ValueError("division count does not match source segment count")
    sim_pps = int(np.sum(divisions)) + 1
    sim = np.empty((n_strands, sim_pps, 3), dtype=np.float32)
    out_i = 0
    for source_i, step_count in enumerate(divisions):
        step_count = int(step_count)
        start = points[:, source_i, :]
        end = points[:, source_i + 1, :]
        for step_i in range(step_count):
            t = float(step_i) / float(step_count)
            sim[:, out_i, :] = start * (1.0 - t) + end * t
            out_i += 1
    sim[:, out_i, :] = points[:, -1, :]
    return sim


def _source_sample_indices(divisions: np.ndarray) -> np.ndarray:
    indices = [0]
    current = 0
    for count in divisions:
        current += int(count)
        indices.append(current)
    return np.asarray(indices, dtype=np.int32)


def _resample_uniform_chain(points: np.ndarray,
                            target_segment_length_m: float,
                            interpolation_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    divisions = _segment_divisions(points, target_segment_length_m, interpolation_steps)
    sim = _resample_chain_with_divisions(points, divisions)
    source_indices = _source_sample_indices(divisions)
    return sim, divisions, source_indices


def _restore_source_points(sim_points: np.ndarray,
                           source_indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(sim_points[:, source_indices, :], dtype=np.float32)


def _memory_weights(sim_points: np.ndarray, memory_height_m: float) -> np.ndarray:
    """Mark the start-frame upper groom shape that should pull back to style."""
    weights = (sim_points[:, :, 2] >= float(memory_height_m)).astype(np.float32)
    weights[:, 0] = 0.0
    return np.ascontiguousarray(weights, dtype=np.float32)


def prepare_gravity_sim(curves_obj, collider_obj, points_per_strand: int,
                        radius_m: float, sort_axis: str = "Z",
                        cylinder_length_m: float = si.DEFAULT_CYLINDER_LENGTH
                        ) -> dict:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if isinstance(collider_obj, (list, tuple)):
        if not any(obj is not None and obj.type == "MESH" for obj in collider_obj):
            raise ValueError("expected at least one Mesh collider")
    elif collider_obj is None or collider_obj.type != "MESH":
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
                          collider_max_move_m: float = 0.001,
                          target_segment_length_m: float = si.DEFAULT_CYLINDER_LENGTH,
                          interpolation_steps: int = 1,
                          propagation_length_m: float = 0.5,
                          memory_height_m: float = 1.5,
                          memory_strength: float = 0.55,
                          bake_mode: str = "FINAL") -> GravityBakeStats:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if isinstance(collider_obj, (list, tuple)):
        if not any(obj is not None and obj.type == "MESH" for obj in collider_obj):
            raise ValueError("expected at least one Mesh collider")
    elif collider_obj is None or collider_obj.type != "MESH":
        raise ValueError("expected one Mesh collider")
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame < start_frame:
        raise ValueError("End Frame must be >= Start Frame")
    bake_mode = str(bake_mode).strip().upper()
    if bake_mode not in {"FINAL", "KEYFRAMES"}:
        bake_mode = "FINAL"

    scene = bpy.context.scene
    original_frame = int(scene.frame_current)
    t0 = time.perf_counter()
    try:
        scene.frame_set(start_frame)
        pps, n_strands = _points_per_strand(curves_obj)
        if pps != int(points_per_strand):
            raise ValueError(f"points_per_strand mismatch: UI={points_per_strand}, data={pps}")

        eval_world, _original_world = si._read_world(curves_obj)
        source = eval_world.reshape(n_strands, pps, 3).astype(np.float32, copy=True)
        sim, divisions, source_indices = _resample_uniform_chain(
            source,
            float(target_segment_length_m),
            int(interpolation_steps),
        )
        seg_lens = np.linalg.norm(sim[:, 1:, :] - sim[:, :-1, :], axis=2).astype(np.float32)
        max_segment_m = float(np.max(seg_lens)) if seg_lens.size else 0.0
        style_weights = _memory_weights(sim, float(memory_height_m))
        memory_points = int(np.count_nonzero(style_weights))
        roots_prev = sim[:, 0, :].copy()

        frames = list(range(start_frame, end_frame + 1))
        baked = {}
        offsets = {}
        max_substeps = 1
        max_root_move = 0.0
        total_hits = 0
        for frame in frames:
            scene.frame_set(frame)
            eval_world, original_world = si._read_world(curves_obj)
            offsets[frame] = eval_world - original_world
            eval_source = eval_world.reshape(n_strands, pps, 3).astype(np.float32, copy=True)
            eval_roots = eval_source[:, 0, :].copy()
            style_targets = _resample_chain_with_divisions(eval_source, divisions)
            vertices, triangles = cuda_collider.mesh_world_triangles(collider_obj)

            if frame == start_frame:
                baked[frame] = _restore_source_points(sim, source_indices).reshape(
                    n_strands * pps, 3
                )
                roots_prev = eval_roots
                continue

            root_moves = np.linalg.norm(eval_roots - roots_prev, axis=1)
            move_m = float(np.max(root_moves)) if len(root_moves) else 0.0
            substeps = max(1, int(math.ceil(move_m * 1000.0)))
            max_substeps = max(max_substeps, substeps)
            max_root_move = max(max_root_move, move_m)

            sim, hit_count = cuda_collider.simulate_chain_frame_arrays(
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
                propagation_length_m=propagation_length_m,
                style_targets_xyz=style_targets,
                style_weights=style_weights,
                style_strength=memory_strength,
            )
            total_hits += int(hit_count)

            roots_prev = eval_roots
            baked[frame] = _restore_source_points(sim, source_indices).reshape(
                n_strands * pps, 3
            )

        if bake_mode == "KEYFRAMES":
            for frame in frames:
                scene.frame_set(frame)
                _write_and_keyframe(curves_obj, baked[frame], frame, offset=offsets[frame])
        else:
            scene.frame_set(end_frame)
            si._write_world_points(curves_obj, baked[end_frame], offset=offsets[end_frame])

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
        total_hits=total_hits,
        interpolation_steps=int(interpolation_steps),
        sim_points_per_strand=int(sim.shape[1]),
        target_segment_cm=float(target_segment_length_m) * 100.0,
        max_segment_mm=max_segment_m * 1000.0,
        propagation_length_cm=float(propagation_length_m) * 100.0,
        memory_height_m=float(memory_height_m),
        memory_strength=float(memory_strength),
        memory_points=memory_points,
        bake_mode=bake_mode,
        elapsed_sec=time.perf_counter() - t0,
    )
