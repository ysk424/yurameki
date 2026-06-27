"""Yurameki -- single-shot physics simulation.

No bake buffer.  No frame_change_post handler.  No mode state.
The Simulate operator calls run_simulation() directly; it blocks while
running N steps of Warp CUDA XPBD and writes the result back to the Curves
object.

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
                   scene, protected_indices=None) -> str:
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
    dt = float(scene.render.fps_base) / float(scene.render.fps)

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

    print(f'[yurameki/sim] {n_steps} steps, {n_strands} strands, '
          f'ke={SPRING_KE:.4g}, damping={DAMPING:.4g}')

    for _step in range(n_steps):
        solver.set_positions_velocities(curr_world, curr_vel)
        new_root_world = curr_world[root_indices]
        sim_out = solver.run_frame(
            dt                = dt,
            n_substeps        = SUBSTEPS,
            n_iter            = ITERATIONS,
            gravity           = GRAVITY,
            new_root_world    = new_root_world,
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
        curr_vel   = solver.get_velocities_numpy()
        curr_world = sim_out
        if prot_init is not None:
            curr_world[prot_mask] = prot_init
            curr_vel[prot_mask]   = 0.0

    _write_world(obj, curr_world, offset=offset_w)
    print(f'[yurameki/sim] done')
    return f'OK: {n_steps} steps, {n_strands} strands'
