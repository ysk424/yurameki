"""Yurameki -- single-shot physics simulation.

No bake buffer.  No frame_change_post handler.  No mode state.
The Simulate operator calls run_simulation() directly; it blocks while
running N steps of Taichi XPBD and writes the result back to the Curves
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
COMPUTE_BACKEND        = 'CUDA'
COLLISION_MARGIN       = 0.0005
COLLISION_SEARCH       = 0.003
POST_COLLISION_ITERATIONS = 4
ROOT_OFFSET            = 0.001


def _body_bvh(body_name):
    from . import _sim_taichi
    return _sim_taichi.build_body_bvh(body_name)


def condition_to_collider(world_pts, body_name, offset=ROOT_OFFSET, pps=POINTS_PER_STRAND):
    """Translate whole strands outside the Body collider without changing shape.

    Tokoya guarantees this at plant time by creating the Head Mask 1.0 mm outside
    the Body surface. Yurameki accepts external grooms, so it must restore the
    same precondition before the XPBD solver measures rest lengths. Solver and
    collision kernels are intentionally unchanged.

    Important: do NOT project every point independently. That bends the strand
    before rest lengths are measured. Instead find the deepest/nearest violation
    in each strand and apply one translation vector to the whole strand.
    """
    from mathutils import Vector
    bvh = _body_bvh(body_name)
    if bvh is None:
        return world_pts, 0, 0
    out = world_pts.copy()
    pushed = 0
    roots_pushed = 0
    if len(out) % pps:
        return out, 0, 0
    n_strands = len(out) // pps
    for strand in range(n_strands):
        base = strand * pps
        best_delta = None
        best_amount = 0.0
        root_anchor_touched = False
        for k in range(pps):
            i = base + k
            pv = Vector(out[i].tolist())
            loc, normal, _, _ = bvh.find_nearest(pv)
            if loc is None:
                continue
            normal = normal.normalized()
            signed = (pv - loc).dot(normal)
            amount = offset - signed
            if amount > best_amount:
                best_amount = amount
                best_delta = np.array(normal * amount, dtype=np.float32)
            if k < 2 and amount > 0.0:
                root_anchor_touched = True
        if best_delta is not None and best_amount > 0.0:
            out[base:base + pps] += best_delta
            pushed += pps
            if root_anchor_touched:
                roots_pushed += 2
    return out, pushed, roots_pushed


def condition_curve_to_collider(obj, body_name, scene=None, offset=ROOT_OFFSET,
                                pps=POINTS_PER_STRAND, max_passes=3):
    """Persistently condition a Curves object, then re-read evaluated positions.

    The old one-shot approach only conditioned the solver's private array. The
    evaluated Curves object still supplied buried roots on the next frame, and
    `_write_world(..., offset)` used a stale modifier offset. This function writes
    the conditioned target back to the original curve, updates the depsgraph, then
    re-measures the evaluated/original relationship before the solver starts.
    """
    if obj is None or obj.type != 'CURVES':
        return None, None, 0, 0
    attr = obj.data.attributes.get('position')
    if attr is None or len(attr.data) == 0:
        return None, None, 0, 0
    n_total = len(attr.data)
    total_pushed = 0
    total_roots = 0
    scene = scene or bpy.context.scene
    eval_w = orig_w = None
    for _ in range(max(1, int(max_passes))):
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(dg)
        eval_w = _read_world(obj_eval.data, n_total, obj_eval.matrix_world)
        orig_w = _read_world(obj.data, n_total, obj.matrix_world)
        if eval_w is None or orig_w is None:
            return None, None, total_pushed, total_roots
        conditioned, pushed, roots = condition_to_collider(
            eval_w, body_name, offset, pps
        )
        total_pushed += pushed
        total_roots += roots
        if pushed == 0:
            break
        _write_world(obj, conditioned, offset=eval_w - orig_w)
        if scene is not None:
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    eval_w = _read_world(obj_eval.data, n_total, obj_eval.matrix_world)
    orig_w = _read_world(obj.data, n_total, obj.matrix_world)
    return eval_w, orig_w, total_pushed, total_roots



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

    eval_w, orig_w, n_pushed, n_roots = condition_curve_to_collider(
        obj, BODY_COLLISION_TARGET, scene, ROOT_OFFSET, POINTS_PER_STRAND
    )
    if eval_w is None or orig_w is None:
        return 'ERROR: could not read world positions'
    if n_pushed:
        print(f'[yurameki/sim] conditioned {n_pushed} points '
              f'({n_roots} root anchors) to {ROOT_OFFSET * 1000:.2f} mm '
              f'outside {BODY_COLLISION_TARGET!r}')
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
        from . import _sim_taichi
        cls    = _sim_taichi.get_solver_class(COMPUTE_BACKEND)
        solver = cls(
            n_total         = n_total,
            n_strands       = n_strands,
            pps             = POINTS_PER_STRAND,
            init_pos        = curr_world,
            particle_mass   = PARTICLE_MASS,
            bending_enabled = BENDING_ENABLED,
        )
    except Exception as exc:
        return f'ERROR: Taichi solver build failed: {exc!r}'

    root_mask = np.zeros(n_total, dtype=bool)
    root_mask[root_indices]     = True
    root_mask[root_indices + 1] = True

    from . import _sim_taichi as _st
    body_bvh = None
    warp_collision = None
    if COMPUTE_BACKEND == 'CUDA':
        try:
            from ._collision_warp import WarpBodyCollider
            warp_collision = WarpBodyCollider(
                body_name=BODY_COLLISION_TARGET,
                n_total=n_total,
                points_per_strand=POINTS_PER_STRAND,
                margin=COLLISION_MARGIN,
                search_distance=COLLISION_SEARCH,
            )
            print('[yurameki/sim] NVIDIA Warp CUDA collision enabled')
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
            print(
                '[yurameki/sim] Warp collision unavailable; '
                f'using Python BVH: {exc!r}'
            )
    if warp_collision is None:
        body_bvh = _st.build_body_bvh(BODY_COLLISION_TARGET)
        if body_bvh is None:
            print(
                f'[yurameki/sim] WARNING: BVH build failed for '
                f'{BODY_COLLISION_TARGET!r}'
            )

    from mathutils import Vector

    collision_stats = {
        "sweep": 0, "near": 0, "segment": 0, "velocity": 0,
        "reconcile": 0,
    }

    def _body_fn(pos_np, pred_np, vel_np,
                 allow_sweep=True,
                 final_cleanup=False,
                 _bvh=body_bvh, _mask=root_mask):
        if _bvh is None:
            return
        normals = np.zeros_like(pred_np)
        contacted = np.zeros(n_total, dtype=bool)

        # Continuous point collision: sweep old -> predicted position.
        for i in range(n_total):
            if _mask[i]:
                vel_np[i] = 0.0
                continue
            p0 = Vector(pos_np[i].tolist())
            p1 = Vector(pred_np[i].tolist())
            delta = p1 - p0
            length = delta.length
            hit = False
            if allow_sweep and length > 1e-9:
                loc, normal, _, dist = _bvh.ray_cast(
                    p0, delta / length, length
                )
                if (
                    loc is not None
                    and dist <= length
                    and delta.dot(normal) < 0.0
                ):
                    normal.normalize()
                    corrected = loc + normal * COLLISION_MARGIN
                    pred_np[i] = corrected
                    normals[i] = normal
                    contacted[i] = True
                    collision_stats["sweep"] += 1
                    hit = True
            if not hit:
                point = Vector(pred_np[i].tolist())
                loc, normal, _, dist = _bvh.find_nearest(point)
                if loc is not None and dist < COLLISION_SEARCH:
                    normal.normalize()
                    signed = (point - loc).dot(normal)
                    if signed < COLLISION_MARGIN:
                        corrected = loc + normal * COLLISION_MARGIN
                        pred_np[i] = corrected
                        normals[i] = normal
                        contacted[i] = True
                        collision_stats["near"] += 1

        # A polyline edge can cross the body while both endpoint particles
        # remain outside. Constrain every strand segment as well.
        cleanup_passes = 4 if final_cleanup else 1
        for _ in range(cleanup_passes):
            for strand in range(n_strands):
                base = strand * POINTS_PER_STRAND
                for segment in range(POINTS_PER_STRAND - 1):
                    i = base + segment
                    j = i + 1
                    p0 = Vector(pred_np[i].tolist())
                    p1 = Vector(pred_np[j].tolist())
                    delta = p1 - p0
                    length = delta.length
                    if length < 1e-9:
                        continue
                    loc, normal, _, dist = _bvh.ray_cast(
                        p0, delta / length, length
                    )
                    if (
                        loc is None
                        or not (1e-6 < dist < length - 1e-6)
                    ):
                        continue
                    normal.normalize()
                    target = loc + normal * COLLISION_MARGIN
                    if final_cleanup:
                        if not _mask[j]:
                            pred_np[j] = target
                            normals[j] = normal
                            contacted[j] = True
                            collision_stats["segment"] += 1
                        continue
                    correction = np.array(
                        target - loc, dtype=np.float32
                    )
                    fraction = dist / length
                    wi = 0.0 if _mask[i] else 1.0
                    wj = 0.0 if _mask[j] else 1.0
                    denom = wi * (1.0 - fraction) ** 2 + wj * fraction ** 2
                    if denom <= 1e-12:
                        continue
                    if wi > 0.0:
                        scale_i = (1.0 - fraction) * wi / denom
                        pred_np[i] += correction * scale_i
                        normals[i] = normal
                        contacted[i] = True
                    if wj > 0.0:
                        scale_j = fraction * wj / denom
                        pred_np[j] += correction * scale_j
                        normals[j] = normal
                        contacted[j] = True
                    collision_stats["segment"] += 1
                    if not allow_sweep:
                        collision_stats["reconcile"] += 1

        # Remove only inward normal velocity. The collision displacement is
        # not part of vel_np, preventing artificial bounce impulses.
        for i in np.nonzero(contacted)[0]:
            normal = normals[i]
            normal_speed = float(np.dot(vel_np[i], normal))
            if normal_speed < 0.0:
                vel_np[i] -= normal * normal_speed
                collision_stats["velocity"] += 1

    collision_fn = warp_collision if warp_collision is not None else _body_fn

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

    # Final safety audit: spring reconciliation can leave a small number of
    # distal edge crossings. Resolve only those residual crossings without
    # feeding the displacement back into velocity.
    for _ in range(8):
        before = collision_stats["segment"]
        collision_fn(
            curr_world, curr_world, curr_vel,
            allow_sweep=False, final_cleanup=True,
        )
        if collision_stats["segment"] == before:
            break

    _write_world(obj, curr_world, offset=offset_w)
    print(
        '[yurameki/sim] collision totals: '
        f'sweep={collision_stats["sweep"]}, '
        f'near={collision_stats["near"]}, '
        f'segment={collision_stats["segment"]}, '
        f'reconcile={collision_stats["reconcile"]}, '
        f'velocity={collision_stats["velocity"]}'
    )
    print(f'[yurameki/sim] done')
    return f'OK: {n_steps} steps, {n_strands} strands'
