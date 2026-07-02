"""Blender <-> solver bridge for the cylinder-based CUDA solver path.

The first milestone is deliberately CPU/numpy only: it proves that Curves hair
can be read, decomposed into fixed-length cylinders, ordered deterministically,
and reconstructed back to the original Blender point count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import bpy
import numpy as np


DEFAULT_CYLINDER_LENGTH = 0.01  # meters
DEFAULT_AXIS_STEP = 0.0005      # meters
DEFAULT_TIP_BACKWARD = 0.03     # meters


@dataclass
class CylinderModel:
    world: np.ndarray
    points_per_strand: int
    roots: np.ndarray
    dirs: np.ndarray
    lengths: np.ndarray
    strand_index: np.ndarray
    cylinder_index: np.ndarray
    strand_offsets: np.ndarray
    source_distances: np.ndarray
    order: np.ndarray
    sort_axis: str

    @property
    def n_strands(self) -> int:
        return int(len(self.strand_offsets) - 1)

    @property
    def n_cylinders(self) -> int:
        return int(len(self.roots))


def _axis_column(axis: str) -> tuple[int, bool]:
    axis = str(axis).strip().upper()
    descending = axis.startswith("-")
    if descending:
        axis = axis[1:]
    if axis not in {"X", "Y", "Z"}:
        axis = "Z"
    return "XYZ".index(axis), descending


def _stable_order(roots: np.ndarray, strand_index: np.ndarray,
                  cylinder_index: np.ndarray, axis: str) -> np.ndarray:
    col, descending = _axis_column(axis)
    key = -roots[:, col] if descending else roots[:, col]
    return np.lexsort((cylinder_index, strand_index, key)).astype(np.int32)


def _read_world(curves_obj):
    attr = curves_obj.data.attributes.get("position")
    if attr is None:
        raise ValueError("Curves has no position attribute")
    n_total = len(attr.data)
    if n_total == 0:
        raise ValueError("Curves has no points")
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = curves_obj.evaluated_get(dg)
    world = _read_world_points(obj_eval.data, n_total, obj_eval.matrix_world)
    original = _read_world_points(curves_obj.data, n_total, curves_obj.matrix_world)
    if world is None or original is None:
        raise ValueError("Could not read world-space Curves positions")
    return world.astype(np.float32, copy=False), original.astype(np.float32, copy=False)


def _read_world_points(data_owner, n_total: int, matrix_world):
    attr = data_owner.attributes.get("position")
    if attr is None or len(attr.data) != n_total:
        return None
    flat = np.zeros(n_total * 3, dtype=np.float32)
    attr.data.foreach_get("vector", flat)
    local_pts = flat.reshape(n_total, 3)
    mw = np.array(matrix_world, dtype=np.float32)
    lh = np.column_stack([local_pts, np.ones(n_total, dtype=np.float32)])
    return (lh @ mw.T)[:, :3].astype(np.float32, copy=True)


def _write_world_points(curves_obj, world_pts: np.ndarray, offset=None) -> None:
    n_total = len(world_pts)
    write_pts = world_pts if offset is None else world_pts - offset
    mw_inv = np.array(curves_obj.matrix_world.inverted(), dtype=np.float32)
    wh = np.column_stack([write_pts, np.ones(n_total, dtype=np.float32)])
    local_pts = (wh @ mw_inv.T)[:, :3].astype(np.float32, copy=True)
    attr = curves_obj.data.attributes.get("position")
    if attr is None or len(attr.data) != n_total:
        raise ValueError("Curves position attribute shape changed")
    attr.data.foreach_set("vector", local_pts.ravel())
    curves_obj.data.update_tag()


def _sample_polyline(points: np.ndarray, distances: np.ndarray,
                     target: float) -> np.ndarray:
    if target <= 0.0:
        return points[0].copy()
    if target >= float(distances[-1]):
        return points[-1].copy()
    hi = int(np.searchsorted(distances, target, side="right"))
    lo = max(0, hi - 1)
    span = float(distances[hi] - distances[lo])
    if span <= 1.0e-9:
        return points[lo].copy()
    t = (float(target) - float(distances[lo])) / span
    return (points[lo] * (1.0 - t) + points[hi] * t).astype(np.float32)


def _unit_or(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1.0e-9:
        return fallback.astype(np.float32, copy=True)
    return (v / length).astype(np.float32)


def build_cylinder_model(curves_obj, points_per_strand: int,
                         sort_axis: str = "Z",
                         cylinder_length_m: float = DEFAULT_CYLINDER_LENGTH
                         ) -> CylinderModel:
    if curves_obj is None or curves_obj.type != "CURVES":
        raise ValueError("expected one Curves object")
    if points_per_strand < 2:
        raise ValueError("points_per_strand must be at least 2")
    if cylinder_length_m <= 0.0:
        raise ValueError("cylinder_length_m must be positive")

    world, _original = _read_world(curves_obj)
    n_total = len(world)
    if n_total % points_per_strand != 0:
        raise ValueError(
            f"n_total={n_total} is not divisible by points_per_strand={points_per_strand}"
        )

    strands = world.reshape(-1, points_per_strand, 3)
    roots = []
    dirs = []
    lengths = []
    strand_ids = []
    cylinder_ids = []
    offsets = [0]
    source_distances = np.zeros(
        (len(strands), points_per_strand), dtype=np.float32
    )

    fallback_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for si, strand in enumerate(strands):
        seg = strand[1:] - strand[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        dist = np.concatenate(
            [np.array([0.0], dtype=np.float32), np.cumsum(seg_len).astype(np.float32)]
        )
        source_distances[si] = dist
        total = float(dist[-1])
        n_cyl = max(1, int(np.ceil(max(total, cylinder_length_m) / cylinder_length_m)))

        for ci in range(n_cyl):
            start_d = min(float(ci) * cylinder_length_m, total)
            end_d = min(float(ci + 1) * cylinder_length_m, total)
            start = _sample_polyline(strand, dist, start_d)
            end = _sample_polyline(strand, dist, end_d)
            direction = _unit_or(end - start, fallback_dir)
            roots.append(start)
            dirs.append(direction)
            lengths.append(cylinder_length_m)
            strand_ids.append(si)
            cylinder_ids.append(ci)
        offsets.append(len(roots))

    roots = np.asarray(roots, dtype=np.float32)
    dirs = np.asarray(dirs, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    strand_index = np.asarray(strand_ids, dtype=np.int32)
    cylinder_index = np.asarray(cylinder_ids, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int32)
    order = _stable_order(roots, strand_index, cylinder_index, sort_axis)
    return CylinderModel(
        world=world,
        points_per_strand=points_per_strand,
        roots=roots,
        dirs=dirs,
        lengths=lengths,
        strand_index=strand_index,
        cylinder_index=cylinder_index,
        strand_offsets=offsets,
        source_distances=source_distances,
        order=order,
        sort_axis=sort_axis,
    )


def build_probe_step(curves_obj, points_per_strand: int, sort_axis: str = "Z",
                     target_length_m: float = DEFAULT_CYLINDER_LENGTH,
                     axis_step_m: float = DEFAULT_AXIS_STEP,
                     tip_back_m: float = DEFAULT_TIP_BACKWARD):
    model = build_cylinder_model(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        cylinder_length_m=target_length_m,
    )
    out_roots = np.zeros_like(model.roots)
    out_tips = np.zeros_like(model.roots)
    y_step = np.array([0.0, axis_step_m, 0.0], dtype=np.float32)
    tip_back = np.array([0.0, 0.0, -tip_back_m], dtype=np.float32)

    for si in range(model.n_strands):
        begin = int(model.strand_offsets[si])
        end = int(model.strand_offsets[si + 1])
        current = model.roots[begin].copy()
        for local, ci in enumerate(range(begin, end)):
            out_roots[ci] = current
            desired = current + model.dirs[ci] * model.lengths[ci] + y_step + tip_back
            direction = _unit_or(desired - current, model.dirs[ci])
            tip = current + direction * model.lengths[ci]
            out_tips[ci] = tip
            current = tip

    length_error = np.linalg.norm(out_tips - out_roots, axis=1) - model.lengths
    return {
        "model": model,
        "roots": model.roots,
        "dirs": model.dirs,
        "lengths": model.lengths,
        "strand_index": model.strand_index,
        "cylinder_index": model.cylinder_index,
        "strand_offsets": model.strand_offsets,
        "order": model.order,
        "probe_roots": out_roots,
        "probe_tips": out_tips,
        "length_error_m": length_error.astype(np.float32),
    }


def build_root_pull_fk_step(curves_obj, points_per_strand: int,
                            sort_axis: str = "Z",
                            target_length_m: float = DEFAULT_CYLINDER_LENGTH,
                            root_pull_y_m: float = DEFAULT_AXIS_STEP):
    model = build_cylinder_model(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        cylinder_length_m=target_length_m,
    )
    out_roots = np.zeros_like(model.roots)
    out_tips = np.zeros_like(model.roots)
    pull = np.array([0.0, root_pull_y_m, 0.0], dtype=np.float32)

    max_chain_gap = 0.0
    max_dir_delta = 0.0
    root_displacements = []
    tip_displacements = []

    for si in range(model.n_strands):
        begin = int(model.strand_offsets[si])
        end = int(model.strand_offsets[si + 1])
        current = model.roots[begin] + pull
        root_displacements.append(float(np.linalg.norm(current - model.roots[begin])))

        for ci in range(begin, end):
            out_roots[ci] = current
            direction = _unit_or(model.dirs[ci], np.array([0.0, 0.0, -1.0], dtype=np.float32))
            tip = current + direction * model.lengths[ci]
            out_tips[ci] = tip
            max_dir_delta = max(
                max_dir_delta,
                float(np.linalg.norm(direction - model.dirs[ci])),
            )
            current = tip

        if end - begin > 1:
            gaps = np.linalg.norm(out_roots[begin + 1:end] - out_tips[begin:end - 1], axis=1)
            max_chain_gap = max(max_chain_gap, float(np.max(gaps)))

        if end > begin:
            tip_displacements.append(float(np.linalg.norm(out_tips[end - 1] - (
                model.roots[end - 1] + model.dirs[end - 1] * model.lengths[end - 1]
            ))))

    length_error = np.linalg.norm(out_tips - out_roots, axis=1) - model.lengths
    return {
        "model": model,
        "roots": model.roots,
        "dirs": model.dirs,
        "lengths": model.lengths,
        "strand_index": model.strand_index,
        "cylinder_index": model.cylinder_index,
        "strand_offsets": model.strand_offsets,
        "order": model.order,
        "fk_roots": out_roots,
        "fk_tips": out_tips,
        "length_error_m": length_error.astype(np.float32),
        "max_chain_gap_m": float(max_chain_gap),
        "max_dir_delta": float(max_dir_delta),
        "max_root_displacement_m": float(max(root_displacements) if root_displacements else 0.0),
        "max_tip_displacement_m": float(max(tip_displacements) if tip_displacements else 0.0),
    }


def reconstruct_points(model: CylinderModel, cylinder_roots: np.ndarray,
                       cylinder_tips: np.ndarray) -> np.ndarray:
    out = np.zeros_like(model.world)
    for si in range(model.n_strands):
        begin = int(model.strand_offsets[si])
        end = int(model.strand_offsets[si + 1])
        chain = [cylinder_roots[begin]]
        chain.extend(cylinder_tips[begin:end])
        chain = np.asarray(chain, dtype=np.float32)
        chain_dist = np.concatenate([
            np.array([0.0], dtype=np.float32),
            np.cumsum(model.lengths[begin:end]).astype(np.float32),
        ])
        if len(chain_dist) > 1:
            chain_dist[-1] = max(chain_dist[-1], model.source_distances[si, -1])
        for pi, target in enumerate(model.source_distances[si]):
            out[si * model.points_per_strand + pi] = _sample_polyline(
                chain, chain_dist, float(target)
            )
    return out


def apply_probe_step(curves_obj, points_per_strand: int, sort_axis: str = "Z",
                     target_length_m: float = DEFAULT_CYLINDER_LENGTH,
                     axis_step_m: float = DEFAULT_AXIS_STEP,
                     tip_back_m: float = DEFAULT_TIP_BACKWARD) -> dict:
    probe = build_probe_step(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        target_length_m=target_length_m,
        axis_step_m=axis_step_m,
        tip_back_m=tip_back_m,
    )
    model = probe["model"]
    _eval_world, original_world = _read_world(curves_obj)
    offset = model.world - original_world
    reconstructed = reconstruct_points(model, probe["probe_roots"], probe["probe_tips"])
    _write_world_points(curves_obj, reconstructed, offset=offset)
    return {
        "n_strands": model.n_strands,
        "n_cylinders": model.n_cylinders,
        "max_len_err_mm": float(np.max(np.abs(probe["length_error_m"])) * 1000.0),
    }


def apply_root_pull_fk_step(curves_obj, points_per_strand: int,
                            sort_axis: str = "Z",
                            target_length_m: float = DEFAULT_CYLINDER_LENGTH,
                            root_pull_y_m: float = DEFAULT_AXIS_STEP) -> dict:
    fk = build_root_pull_fk_step(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        target_length_m=target_length_m,
        root_pull_y_m=root_pull_y_m,
    )
    model = fk["model"]
    _eval_world, original_world = _read_world(curves_obj)
    offset = model.world - original_world
    reconstructed = reconstruct_points(model, fk["fk_roots"], fk["fk_tips"])
    _write_world_points(curves_obj, reconstructed, offset=offset)
    return {
        "n_strands": model.n_strands,
        "n_cylinders": model.n_cylinders,
        "max_len_err_mm": float(np.max(np.abs(fk["length_error_m"])) * 1000.0),
        "max_chain_gap_mm": float(fk["max_chain_gap_m"] * 1000.0),
        "max_dir_delta": float(fk["max_dir_delta"]),
        "max_root_displacement_mm": float(fk["max_root_displacement_m"] * 1000.0),
        "max_tip_displacement_mm": float(fk["max_tip_displacement_m"] * 1000.0),
    }


def export_probe_data(curves_obj, path: str, **kwargs) -> str:
    data = build_probe_step(curves_obj, **kwargs)
    model = data["model"]
    np.savez_compressed(
        path,
        order=data["order"],
        roots=data["roots"].astype(np.float32, copy=False),
        dirs=data["dirs"].astype(np.float32, copy=False),
        lengths=data["lengths"].astype(np.float32, copy=False),
        strand_index=data["strand_index"],
        cylinder_index=data["cylinder_index"],
        strand_offsets=data["strand_offsets"],
        probe_roots=data["probe_roots"].astype(np.float32, copy=False),
        probe_tips=data["probe_tips"].astype(np.float32, copy=False),
        length_error_m=data["length_error_m"].astype(np.float32, copy=False),
        sort_axis=np.array([model.sort_axis]),
        n_strands=np.array([model.n_strands], dtype=np.int32),
        n_cylinders=np.array([model.n_cylinders], dtype=np.int32),
        points_per_strand=np.array([model.points_per_strand], dtype=np.int32),
    )
    return os.path.abspath(path)


def export_probe_json(curves_obj, path: str, **kwargs) -> str:
    data = build_probe_step(curves_obj, **kwargs)
    model = data["model"]
    payload = {
        "n_strands": model.n_strands,
        "n_cylinders": model.n_cylinders,
        "points_per_strand": model.points_per_strand,
        "sort_axis": model.sort_axis,
        "order_head": data["order"][:32].tolist(),
        "order_tail": data["order"][-32:].tolist(),
        "max_len_err_mm": float(np.max(np.abs(data["length_error_m"])) * 1000.0),
        "tip_back_m": float(kwargs.get("tip_back_m", DEFAULT_TIP_BACKWARD)),
        "axis_step_m": float(kwargs.get("axis_step_m", DEFAULT_AXIS_STEP)),
        "target_length_m": float(kwargs.get("target_length_m", DEFAULT_CYLINDER_LENGTH)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def gravity_direction_for_step(step_index: int, blend_steps: int) -> np.ndarray:
    """Blend startup gravity from back direction (+Y) to normal gravity (-Z)."""
    if blend_steps <= 0:
        t = 1.0
    else:
        t = min(1.0, max(0.0, float(step_index) / float(blend_steps)))
    start = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    end = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return _unit_or(start * (1.0 - t) + end * t, end)


def build_directional_gravity_fk_step(curves_obj, points_per_strand: int,
                                      sort_axis: str = "Z",
                                      target_length_m: float = DEFAULT_CYLINDER_LENGTH,
                                      gravity_step_m: float = 0.001,
                                      step_index: int = 0,
                                      gravity_blend_steps: int = 12):
    model = build_cylinder_model(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        cylinder_length_m=target_length_m,
    )
    out_roots = np.zeros_like(model.roots)
    out_tips = np.zeros_like(model.roots)
    gravity_dir = gravity_direction_for_step(step_index, gravity_blend_steps)
    gravity = gravity_dir * float(gravity_step_m)

    max_chain_gap = 0.0
    max_tip_displacement = 0.0

    for si in range(model.n_strands):
        begin = int(model.strand_offsets[si])
        end = int(model.strand_offsets[si + 1])
        current = model.roots[begin].copy()

        for ci in range(begin, end):
            out_roots[ci] = current
            desired = current + model.dirs[ci] * model.lengths[ci] + gravity
            direction = _unit_or(desired - current, model.dirs[ci])
            tip = current + direction * model.lengths[ci]
            out_tips[ci] = tip
            original_tip = model.roots[ci] + model.dirs[ci] * model.lengths[ci]
            max_tip_displacement = max(
                max_tip_displacement,
                float(np.linalg.norm(tip - original_tip)),
            )
            current = tip

        if end - begin > 1:
            gaps = np.linalg.norm(out_roots[begin + 1:end] - out_tips[begin:end - 1], axis=1)
            max_chain_gap = max(max_chain_gap, float(np.max(gaps)))

    length_error = np.linalg.norm(out_tips - out_roots, axis=1) - model.lengths
    return {
        "model": model,
        "roots": out_roots,
        "tips": out_tips,
        "length_error_m": length_error.astype(np.float32),
        "max_chain_gap_m": float(max_chain_gap),
        "max_tip_displacement_m": float(max_tip_displacement),
        "gravity_dir": gravity_dir.astype(np.float32),
    }


def apply_directional_gravity_fk_step(curves_obj, points_per_strand: int,
                                      sort_axis: str = "Z",
                                      target_length_m: float = DEFAULT_CYLINDER_LENGTH,
                                      gravity_step_m: float = 0.001,
                                      step_index: int = 0,
                                      gravity_blend_steps: int = 12) -> dict:
    step = build_directional_gravity_fk_step(
        curves_obj,
        points_per_strand=points_per_strand,
        sort_axis=sort_axis,
        target_length_m=target_length_m,
        gravity_step_m=gravity_step_m,
        step_index=step_index,
        gravity_blend_steps=gravity_blend_steps,
    )
    model = step["model"]
    _eval_world, original_world = _read_world(curves_obj)
    offset = model.world - original_world
    reconstructed = reconstruct_points(model, step["roots"], step["tips"])
    _write_world_points(curves_obj, reconstructed, offset=offset)
    gravity_dir = step["gravity_dir"]
    return {
        "n_strands": model.n_strands,
        "n_cylinders": model.n_cylinders,
        "max_len_err_mm": float(np.max(np.abs(step["length_error_m"])) * 1000.0),
        "max_chain_gap_mm": float(step["max_chain_gap_m"] * 1000.0),
        "max_tip_displacement_mm": float(step["max_tip_displacement_m"] * 1000.0),
        "gravity_dir": (
            float(gravity_dir[0]),
            float(gravity_dir[1]),
            float(gravity_dir[2]),
        ),
    }
