"""Manual comb repair operators for baked Yurameki frames."""
from __future__ import annotations

import math
from dataclasses import dataclass

import bpy
import numpy as np

from . import _recording
from . import _world_passthrough as _wp


POINTS_PER_STRAND = 9
COMB1_THRESHOLD_MM = 20.0
COMB2_ANGLE_RAD = 0.5
NEIGHBOUR_SCORE_COUNT = 16
NEIGHBOUR_POOL_COUNT = 32
NEIGHBOUR_USE_COUNT = 4
DISPLAY_WRITE_PASSES = 8
DISPLAY_RESIDUAL_EPS = 1.0e-5


@dataclass
class CombResult:
    ok: bool
    message: str
    selected: int = 0
    repaired: int = 0
    skipped: int = 0
    display_passes: int = 0
    display_residual_mm: float = 0.0


def _find_curves_obj():
    objects = [obj for obj in bpy.data.objects if obj.type == "CURVES"]
    return objects[0] if len(objects) == 1 else None


def _tail_max_angles(positions: np.ndarray) -> np.ndarray:
    points = positions.reshape((-1, POINTS_PER_STRAND, 3)).astype(
        np.float64, copy=False
    )
    segments = points[:, 1:, :] - points[:, :-1, :]
    lengths = np.linalg.norm(segments, axis=2)
    dirs = segments / (lengths[:, :, None] + 1.0e-12)
    bends = []
    for point_index in range(1, POINTS_PER_STRAND - 1):
        dot = np.sum(
            dirs[:, point_index - 1, :] * dirs[:, point_index, :],
            axis=1,
        )
        bends.append(np.arccos(np.clip(dot, -1.0, 1.0)))
    bend_array = np.stack(bends, axis=1)
    return bend_array[:, [4, 5, 6]].max(axis=1)


def _basis_and_neighbours(obj, n_strands: int) -> tuple[np.ndarray, np.ndarray]:
    uv_attr = obj.data.attributes.get("surface_uv_coordinate")
    if uv_attr is not None and len(uv_attr.data) == n_strands:
        flat = np.zeros(n_strands * 2, dtype=np.float32)
        uv_attr.data.foreach_get("vector", flat)
        basis = flat.reshape(n_strands, 2).astype(np.float64)
    else:
        raise RuntimeError("Curves has no surface_uv_coordinate attribute")

    diff = basis[:, None, :] - basis[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    pool = min(NEIGHBOUR_POOL_COUNT, max(1, n_strands - 1))
    neighbours = np.argpartition(dist2, pool, axis=1)[:, :pool]
    order = np.argsort(np.take_along_axis(dist2, neighbours, axis=1), axis=1)
    neighbours = np.take_along_axis(neighbours, order, axis=1)
    return basis, neighbours


def _comb1_scores(positions: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    points = positions.reshape((-1, POINTS_PER_STRAND, 3)).astype(
        np.float64, copy=False
    )
    score_count = min(NEIGHBOUR_SCORE_COUNT, neighbours.shape[1])
    score_neighbours = neighbours[:, :score_count]
    point_indices = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    neighbour_points = points[
        score_neighbours[:, :, None],
        point_indices[None, None, :],
        :,
    ]
    self_points = points[:, None, point_indices, :]
    distances = np.linalg.norm(self_points - neighbour_points, axis=3) * 1000.0
    return np.max(np.median(distances, axis=1), axis=1)


def _repair_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    bad_mask: np.ndarray,
    basis: np.ndarray,
    neighbours: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    points = positions.reshape((-1, POINTS_PER_STRAND, 3)).astype(
        np.float64, copy=True
    )
    out = points.copy()
    out_vel = velocities.reshape((-1, POINTS_PER_STRAND, 3)).copy()

    repaired = 0
    skipped = 0
    for strand in np.nonzero(bad_mask)[0]:
        valid = [int(index) for index in neighbours[strand] if not bad_mask[int(index)]]
        if len(valid) < 2:
            skipped += 1
            continue
        valid = valid[: min(NEIGHBOUR_USE_COUNT, len(valid))]
        distances = np.linalg.norm(basis[valid] - basis[strand], axis=1) + 1.0e-12
        weights = 1.0 / distances
        weights = weights / weights.sum()
        relative = points[valid, 1:, :] - points[valid, 0:1, :]
        target = points[strand, 0:1, :] + (
            relative * weights[:, None, None]
        ).sum(axis=0)
        out[strand, 1:, :] = target
        out_vel[strand, :, :] = 0.0
        repaired += 1

    return (
        out.reshape(positions.shape).astype(np.float32),
        out_vel.reshape(velocities.shape).astype(np.float32),
        repaired,
        skipped,
    )


def _display_cached_frame_iterative(obj, frame: int) -> tuple[int, float]:
    manager = _recording.manager
    cached = manager.frames.get(int(frame))
    if cached is None:
        return 0, 0.0
    desired = cached[0].astype(np.float64, copy=False)
    n_total = int(desired.shape[0])
    residual_mm = 0.0
    for pass_index in range(1, DISPLAY_WRITE_PASSES + 1):
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(dg)
        eval_world = _wp._read_world(obj_eval.data, n_total, obj_eval.matrix_world)
        orig_world = _wp._read_world(obj.data, n_total, obj.matrix_world)
        if eval_world is None or orig_world is None:
            return pass_index - 1, residual_mm
        _wp._write_world(obj, desired, offset=eval_world - orig_world)
        bpy.context.view_layer.update()

        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(dg)
        eval_after = _wp._read_world(
            obj_eval.data, n_total, obj_eval.matrix_world
        )
        if eval_after is None:
            return pass_index, residual_mm
        residual = np.linalg.norm(eval_after - desired, axis=1)
        residual_mm = float(residual.max() * 1000.0)
        if float(residual.max()) <= DISPLAY_RESIDUAL_EPS:
            return pass_index, residual_mm
    return DISPLAY_WRITE_PASSES, residual_mm


def _comb_current_frame(make_bad_mask) -> CombResult:
    obj = _find_curves_obj()
    if obj is None:
        return CombResult(False, "Need exactly one Curves object")

    manager = _recording.manager
    frame = int(bpy.context.scene.frame_current)
    cached = manager.frames.get(frame)
    if cached is None:
        return CombResult(False, f"No baked cache for frame {frame}")

    positions = cached[0].astype(np.float32, copy=True)
    velocities = cached[1].astype(np.float32, copy=True)
    if positions.ndim != 2 or positions.shape[1] != 3:
        return CombResult(False, "Invalid baked position cache")
    if len(positions) % POINTS_PER_STRAND:
        return CombResult(False, "Point count is not divisible by strand size")

    n_strands = len(positions) // POINTS_PER_STRAND
    try:
        basis, neighbours = _basis_and_neighbours(obj, n_strands)
    except Exception as exc:
        return CombResult(False, f"Neighbour build failed: {exc!r}")

    bad_mask = make_bad_mask(positions, neighbours)
    selected = int(np.sum(bad_mask))
    if selected == 0:
        passes, residual = _display_cached_frame_iterative(obj, frame)
        return CombResult(
            True,
            f"No strands selected on frame {frame}",
            display_passes=passes,
            display_residual_mm=residual,
        )

    fixed_positions, fixed_velocities, repaired, skipped = _repair_positions(
        positions, velocities, bad_mask, basis, neighbours
    )
    manager.frames[frame] = (fixed_positions, fixed_velocities)
    manager.dirty = True
    passes, residual = _display_cached_frame_iterative(obj, frame)
    return CombResult(
        True,
        (
            f"Frame {frame}: selected {selected}, repaired {repaired}, "
            f"skipped {skipped}"
        ),
        selected=selected,
        repaired=repaired,
        skipped=skipped,
        display_passes=passes,
        display_residual_mm=residual,
    )


def comb_1_current_frame() -> CombResult:
    def make_bad_mask(positions, neighbours):
        scores = _comb1_scores(positions, neighbours)
        return scores > COMB1_THRESHOLD_MM

    return _comb_current_frame(make_bad_mask)


def comb_2_current_frame() -> CombResult:
    def make_bad_mask(positions, _neighbours):
        return _tail_max_angles(positions) >= COMB2_ANGLE_RAD

    return _comb_current_frame(make_bad_mask)


def comb_3_current_frame() -> CombResult:
    return CombResult(False, "Comb 3 is not implemented yet")
