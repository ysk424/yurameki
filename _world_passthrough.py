"""Yurameki -- fixed-frame relaxation simulation.

No bake buffer.  No frame_change_post handler.  No mode state.
The Simulate operator calls run_simulation() directly; it blocks while
running N fixed animation frames of Warp CUDA XPBD and writes the final result
back to the Curves object.

Modifier-offset compensation (same technique as Katsura):
  The Deform Curves on Surface modifier applies a per-point world-space
  offset that depends on the rig current pose.  We read both original
  and evaluated world positions, measure the offset once, and write:
  original <- sim_out - offset.  The modifier then reconstructs
  evaluated = written + offset = sim_out.

Parameters (module-level globals, written by __init__.py before each call):
  SPRING_KE, DAMPING, PARTICLE_MASS, GRAVITY
  ITERATIONS, SUBSTEPS
  BENDING_ENABLED, ROOT_BENDING_KE, BENDING_KE
  BODY_COLLISION_TARGET
"""
from __future__ import annotations

import bpy
import numpy as np
from mathutils.kdtree import KDTree

POINTS_PER_STRAND = 9  # must match _recording.POINTS_PER_STRAND

SPRING_KE              = 1e4
DAMPING                = 0.05
PARTICLE_MASS          = 1.0
GRAVITY                = (0.0, 0.0, -9.81)
ITERATIONS             = 10
SUBSTEPS               = 1
BENDING_ENABLED        = True
ROOT_BENDING_KE        = 2000.0
BENDING_KE             = 10.0
ANGLE_LIMIT_ENABLED    = True
ANGLE_LIMIT_RAD        = 1.0
ANGLE_LIMIT_KE         = 1.0e6
BODY_COLLISION_TARGET  = 'CC_Base_Body'
CLOTH_COLLISION_TARGET = ''
COLLISION_MARGIN       = 0.0005
COLLISION_SEARCH       = 0.003
POST_COLLISION_ITERATIONS = 4


def collision_target_names():
    names = [BODY_COLLISION_TARGET]
    cloth = str(CLOTH_COLLISION_TARGET).strip()
    if cloth:
        names.append(cloth)
    return names


def _read_world(data_owner, n_total: int, matrix_world) -> 'np.ndarray | None':
    attr = data_owner.attributes.get('position')
    if attr is None or len(attr.data) != n_total:
        return None
    flat = np.zeros(n_total * 3, dtype=np.float32)
    attr.data.foreach_get('vector', flat)
    local_pts = flat.reshape(n_total, 3)
    mw = np.array(matrix_world, dtype=np.float32)
    lh = np.column_stack([local_pts, np.ones(n_total, dtype=np.float32)])
    return (lh @ mw.T)[:, :3].astype(np.float32, copy=True)


def _write_world(obj, world_pts: np.ndarray,
                 offset: 'np.ndarray | None' = None) -> None:
    n         = len(world_pts)
    write_pts = world_pts if offset is None else world_pts - offset
    mw_inv    = np.array(obj.matrix_world.inverted(), dtype=np.float32)
    wh        = np.column_stack([write_pts, np.ones(n, dtype=np.float32)])
    local_pts = (wh @ mw_inv.T)[:, :3].astype(np.float32, copy=True)
    attr      = obj.data.attributes.get('position')
    if attr is None or len(attr.data) != n:
        return
    attr.data.foreach_set('vector', local_pts.ravel())
    obj.data.update_tag()


def run_simulation(curves_obj_name: str, n_steps: int,
                   scene, protected_indices=None,
                   frame_interpolation: int = 1) -> str:
    obj = bpy.data.objects.get(curves_obj_name)
    if obj is None or obj.type != 'CURVES':
        return f'ERROR: {curves_obj_name!r} not found or not CURVES'

    attr = obj.data.attributes.get('position')
    if attr is None:
        return 'ERROR: no position attribute on Curves'
    n_total = len(attr.data)
    if n_total == 0:
        return 'ERROR: Curves object has no points'
    if n_total % POINTS_PER_STRAND != 0:
        return (f'ERROR: n_total={n_total} not divisible by '
                f'POINTS_PER_STRAND={POINTS_PER_STRAND}')

    n_strands    = n_total // POINTS_PER_STRAND
    root_indices = np.arange(n_strands, dtype=np.int32) * POINTS_PER_STRAND
    fps = float(scene.render.fps) / float(scene.render.fps_base)
    if fps <= 0.0:
        return 'ERROR: invalid scene FPS'
    interpolation = max(1, int(frame_interpolation))
    dt_subframe = (1.0 / fps) / float(interpolation)

    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    eval_w = _read_world(
        obj_eval.data, n_total, obj_eval.matrix_world
    )
    orig_w = _read_world(
        obj.data, n_total, obj.matrix_world
    )
    if eval_w is None or orig_w is None:
        return 'ERROR: could not read world positions'
    offset_w = eval_w - orig_w

    curr_world = eval_w.copy()

    curr_vel   = np.zeros_like(curr_world)

    # Build frozen mask for protected strands (all points of those strands stay fixed)
    prot_mask = np.zeros(n_total, dtype=bool)
    if protected_indices is not None and len(protected_indices) > 0:
        for si in protected_indices:
            b = int(si) * POINTS_PER_STRAND
            prot_mask[b:b + POINTS_PER_STRAND] = True
    prot_init = eval_w[prot_mask].copy() if prot_mask.any() else None
    n_prot    = int(prot_mask.sum()) // POINTS_PER_STRAND
    if n_prot > 0:
        print(f'[yurameki/sim] {n_prot} strands protected (frozen inside primitive)')

    try:
        from ._sim_warp import WarpXPBDSolver
        solver = WarpXPBDSolver(
            n_total=n_total,
            n_strands=n_strands,
            pps=POINTS_PER_STRAND,
            init_pos=curr_world,
            particle_mass=PARTICLE_MASS,
            bending_enabled=BENDING_ENABLED,
        )
        print('[yurameki/sim] Warp CUDA shared-state solver enabled')
    except Exception as exc:
        return f'ERROR: Warp CUDA solver build failed: {exc!r}'

    try:
        from ._collision_warp import WarpBodyCollider
        warp_collision = WarpBodyCollider(
            collider_names=collision_target_names(),
            n_total=n_total,
            points_per_strand=POINTS_PER_STRAND,
            margin=COLLISION_MARGIN,
            search_distance=COLLISION_SEARCH,
        )
        print('[yurameki/sim] NVIDIA Warp CUDA collision enabled')
    except Exception as exc:
        return f'ERROR: Warp CUDA collision unavailable: {exc!r}'

    collision_fn = warp_collision

    fixed_roots = eval_w[root_indices].copy()
    fixed_point1s = eval_w[root_indices + 1].copy()

    print(f'[yurameki/sim] {n_steps} fixed frames, '
          f'{interpolation} interpolation, {n_strands} strands, '
          f'ke={SPRING_KE:.4g}, damping={DAMPING:.4g}')

    for _frame in range(n_steps):
        for _interp in range(interpolation):
            try:
                warp_collision.update_from_collider_names()
            except Exception as exc:
                return f'ERROR: Warp collision update failed: {exc!r}'
            sim_out = solver.run_frame(
                dt                = dt_subframe,
                n_substeps        = SUBSTEPS,
                n_iter            = ITERATIONS,
                gravity           = GRAVITY,
                new_root_world    = fixed_roots,
                new_point1_world  = fixed_point1s,
                seg_ke            = SPRING_KE,
                root_bend_ke      = ROOT_BENDING_KE,
                bend_ke           = BENDING_KE,
                damping           = DAMPING,
                bending_enabled   = BENDING_ENABLED,
                body_collision_fn = collision_fn,
                post_collision_iterations = POST_COLLISION_ITERATIONS,
                angle_limit_enabled = ANGLE_LIMIT_ENABLED,
                angle_limit_rad     = ANGLE_LIMIT_RAD,
                angle_limit_ke      = ANGLE_LIMIT_KE,
            )
            curr_world = sim_out
            if prot_init is not None:
                curr_world[prot_mask] = prot_init
                solver.set_positions_velocities(curr_world, curr_vel)

    curr_vel = solver.get_velocities_numpy()
    if prot_init is not None:
        curr_world[prot_mask] = prot_init
        curr_vel[prot_mask] = 0.0

    _write_world(obj, curr_world, offset=offset_w)
    print(f'[yurameki/sim] done')
    return f'OK: {n_steps} frames, {interpolation} interpolation, {n_strands} strands'


def _bend_sum(strands: np.ndarray) -> np.ndarray:
    a = strands[:, 1:-1, :] - strands[:, :-2, :]
    b = strands[:, 2:, :] - strands[:, 1:-1, :]
    al = np.linalg.norm(a, axis=2)
    bl = np.linalg.norm(b, axis=2)
    ok = (al > 1.0e-8) & (bl > 1.0e-8)
    dots = np.zeros_like(al)
    dots[ok] = np.sum(a[ok] * b[ok], axis=1) / (al[ok] * bl[ok])
    return np.sum(np.arccos(np.clip(dots, -1.0, 1.0)), axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1.0e-8:
        return np.zeros(3, dtype=np.float32)
    return (v / length).astype(np.float32, copy=False)


def clean_up_current_frame(curves_obj_name: str, scene) -> str:
    obj = bpy.data.objects.get(curves_obj_name)
    if obj is None or obj.type != 'CURVES':
        return f'ERROR: {curves_obj_name!r} not found or not CURVES'

    attr = obj.data.attributes.get('position')
    if attr is None:
        return 'ERROR: no position attribute on Curves'
    n_total = len(attr.data)
    if n_total == 0:
        return 'ERROR: Curves object has no points'
    if n_total % POINTS_PER_STRAND != 0:
        return (f'ERROR: n_total={n_total} not divisible by '
                f'POINTS_PER_STRAND={POINTS_PER_STRAND}')

    n_strands = n_total // POINTS_PER_STRAND
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    eval_w = _read_world(obj_eval.data, n_total, obj_eval.matrix_world)
    orig_w = _read_world(obj.data, n_total, obj.matrix_world)
    if eval_w is None or orig_w is None:
        return 'ERROR: could not read world positions'
    offset_w = eval_w - orig_w

    pps = POINTS_PER_STRAND
    pos = eval_w.reshape(n_strands, pps, 3).copy()
    original = pos.copy()
    roots = pos[:, 0, :].copy()
    rest_len = np.maximum(
        np.linalg.norm(pos[:, 1:, :] - pos[:, :-1, :], axis=2),
        1.0e-6,
    )
    strand_len = np.sum(rest_len, axis=1)

    dist_threshold = 0.150
    x_threshold = 0.080
    bend_threshold = np.deg2rad(360.0)
    max_iter = 5
    k = 32
    min_guides = 5
    radii = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    length_tolerances = (0.20, 0.35, 0.55)

    def _tree():
        tree = KDTree(n_strands)
        for strand, co in enumerate(roots):
            tree.insert(tuple(float(x) for x in co), strand)
        tree.balance()
        return tree

    def _scores(strands):
        tree = _tree()
        bends = _bend_sum(strands)
        dist = np.zeros(n_strands, dtype=np.float32)
        absx = np.zeros(n_strands, dtype=np.float32)
        for strand, root in enumerate(roots):
            found = tree.find_n(tuple(float(x) for x in root), k + 1)
            ids = [
                int(item[1])
                for item in found
                if int(item[1]) != strand
            ][:k]
            if len(ids) < min_guides:
                continue
            median = np.median(strands[np.asarray(ids, dtype=np.int32)], axis=0)
            tail = (strands[strand] - median)[2:]
            if len(tail):
                dist[strand] = float(np.linalg.norm(tail, axis=1).max())
                absx[strand] = float(np.abs(tail[:, 0]).max())
        bad = (
            (dist > dist_threshold)
            | (absx > x_threshold)
            | (bends > bend_threshold)
        )
        combined = np.maximum.reduce([
            dist / dist_threshold,
            absx / x_threshold,
            bends / bend_threshold,
        ])
        return bends, dist, absx, bad, combined

    forced = set()
    skipped = 0
    for _iteration in range(max_iter):
        _bends, _dist, _absx, bad, combined = _scores(pos)
        bad_indices = np.flatnonzero(bad)
        if len(bad_indices) == 0:
            break

        directions = np.asarray(
            [_unit(pos[i, 1] - pos[i, 0]) for i in range(n_strands)],
            dtype=np.float32,
        )
        repaired_this = 0
        for strand in bad_indices[np.argsort(-combined[bad_indices])]:
            root = roots[strand]
            target_len = max(float(strand_len[strand]), 1.0e-6)
            guide_ids = []
            for tolerance in length_tolerances:
                if len(guide_ids) >= min_guides:
                    break
                length_ok = (
                    np.abs(strand_len - target_len) / target_len
                    <= tolerance
                )
                for radius in radii:
                    droot = np.abs(roots - root)
                    mask = (
                        (droot[:, 0] <= radius)
                        & (droot[:, 1] <= radius)
                        & (droot[:, 2] <= radius)
                        & (~bad)
                        & length_ok
                    )
                    ids = np.flatnonzero(mask)
                    ids = ids[ids != strand]
                    if len(ids) > 0:
                        dot = directions[ids] @ directions[strand]
                        filtered = ids[dot >= 0.0]
                        if len(filtered) >= min_guides:
                            ids = filtered
                    if len(ids) > 0:
                        root_dist = np.linalg.norm(roots[ids] - root, axis=1)
                        guide_ids = ids[np.argsort(root_dist)[:32]]
                        if len(guide_ids) >= min_guides:
                            break
            if len(guide_ids) < min_guides:
                skipped += 1
                continue

            rel = pos[guide_ids] - pos[guide_ids, 0:1, :]
            target = root + np.median(rel, axis=0)
            new_curve = target.copy()
            new_curve[0] = pos[strand, 0]
            new_curve[1] = pos[strand, 1]
            for point in range(2, pps):
                v = new_curve[point] - new_curve[point - 1]
                length = float(np.linalg.norm(v))
                if length <= 1.0e-8:
                    v = target[point] - new_curve[point - 1]
                    length = float(np.linalg.norm(v))
                if length > 1.0e-8:
                    new_curve[point] = (
                        new_curve[point - 1]
                        + v / length * rest_len[strand, point - 1]
                    )
            pos[strand] = new_curve
            forced.add(int(strand))
            repaired_this += 1
        if repaired_this == 0:
            break

    _final_bends, _final_dist, _final_absx, final_bad, _combined = _scores(pos)
    out = pos.reshape(n_total, 3)
    _write_world(obj, out, offset=offset_w)

    moved = np.linalg.norm(pos - original, axis=2).max(axis=1)
    print(
        '[yurameki/cleanup] '
        f'forced={len(forced)}, remaining={int(np.count_nonzero(final_bad))}, '
        f'max_move={float(moved.max() * 1000.0):.1f}mm, skipped={skipped}'
    )
    return (
        f'OK: cleanup forced {len(forced)} strands; '
        f'remaining strong outliers {int(np.count_nonzero(final_bad))}'
    )
