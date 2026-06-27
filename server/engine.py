"""Headless hair simulation engine — Blender-independent.

Drives the kinematic hair roots by the head bone's world motion (validated
rigid-follow model) and runs the Warp CUDA XPBD solver from the yurameki
extension. No bpy. Collision is optional and OFF in this first milestone
(body_collision_fn=None); the head-driven gravity/spring sim runs end to end.

Inputs (numpy):
  groom_rest : (n_total, 3)  rest hair world positions, frame `frame_start`.
  head_world : (n_frames, 4, 4)  head bone world matrices, frame_start..end.
  params     : physics dict (seg_ke, damping, mass, gravity, iterations, ...).

Output:
  positions  : (n_frames, n_total, 3)  simulated world positions per frame.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

# Import the solver from the extension root (one level up).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def default_params(root: str = _ROOT) -> dict:
    with open(os.path.join(root, "yurameki_defaults.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {
        "seg_ke": float(d["SPRING_KE"]),
        "root_bend_ke": float(d["ROOT_BENDING_KE"]),
        "bend_ke": float(d["BENDING_KE"]),
        "damping": float(d["DAMPING"]),
        "mass": float(d["PARTICLE_MASS"]),
        "gravity": list(d["GRAVITY"]),
        "iterations": int(d["ITERATIONS"]),
        "substeps": 1,
        "bending_enabled": bool(d["BENDING_ENABLED"]),
        "angle_limit_enabled": bool(d.get("ANGLE_LIMIT_ENABLED", True)),
        "angle_limit_rad": float(d.get("ANGLE_LIMIT_RAD", 1.0)),
        "angle_limit_ke": float(d.get("ANGLE_LIMIT_KE", 1.0e6)),
        "fps": 24.0,
        "fps_base": 1.0,
        "pps": 9,
        "backend": "CUDA",
    }


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.column_stack([pts, np.ones(len(pts))])
    return (h @ T.T)[:, :3]


def _make_solver(params, n_total, n_strands, groom_rest):
    """Create the Warp CUDA solver."""
    kwargs = dict(
        n_total=n_total, n_strands=n_strands, pps=int(params["pps"]),
        init_pos=groom_rest.astype(np.float32),
        particle_mass=params["mass"],
        bending_enabled=params["bending_enabled"],
    )
    from _sim_warp import WarpXPBDSolver
    return WarpXPBDSolver(**kwargs)


def simulate(groom_rest, head_world, params, progress=None,
             collider=None, body_frames=None) -> np.ndarray:
    """Drive hair roots by head motion and run the XPBD solver.

    If ``collider`` (a WarpBodyCollider) and ``body_frames`` (n_frames, nv, 3)
    world-space body vertices are given, body collision is enabled and the
    collider's mesh is refit each frame. Collision is the unchanged Tokoya
    method; here it is fed headless geometry per frame.
    """
    pps = int(params["pps"])
    n_total = groom_rest.shape[0]
    if n_total % pps:
        raise ValueError(f"{n_total} points not divisible by pps={pps}")
    n_strands = n_total // pps
    nframes = head_world.shape[0]

    root_idx = np.arange(n_strands) * pps
    roots1 = groom_rest[root_idx]
    point1_1 = groom_rest[root_idx + 1]
    H1_inv = np.linalg.inv(head_world[0])

    solver = _make_solver(params, n_total, n_strands, groom_rest)

    dt = float(params["fps_base"]) / float(params["fps"])
    gravity = np.asarray(params["gravity"], np.float32)
    post_iters = int(params.get("post_collision_iterations", 4))

    curr = groom_rest.astype(np.float32).copy()
    vel = np.zeros_like(curr)
    out = np.zeros((nframes, n_total, 3), np.float32)
    out[0] = curr  # frame_start = rest pose

    for fi in range(1, nframes):
        T = head_world[fi] @ H1_inv
        roots_f = _apply(T, roots1).astype(np.float32)
        point1_f = _apply(T, point1_1).astype(np.float32)
        body_fn = None
        if collider is not None and body_frames is not None:
            collider.update_mesh(body_frames[fi])
            body_fn = collider
        solver.set_positions_velocities(curr, vel)
        curr = solver.run_frame(
            dt=dt, n_substeps=int(params["substeps"]),
            n_iter=int(params["iterations"]),
            gravity=gravity, new_root_world=roots_f,
            seg_ke=params["seg_ke"], root_bend_ke=params["root_bend_ke"],
            bend_ke=params["bend_ke"], damping=params["damping"],
            bending_enabled=params["bending_enabled"],
            new_point1_world=point1_f, body_collision_fn=body_fn,
            post_collision_iterations=post_iters,
            angle_limit_enabled=params.get("angle_limit_enabled", True),
            angle_limit_rad=params.get("angle_limit_rad", 1.0),
            angle_limit_ke=params.get("angle_limit_ke", 1.0e6),
        )
        vel = solver.get_velocities_numpy()
        out[fi] = curr
        if progress and fi % 25 == 0:
            progress(fi, nframes)
    return out


def _build_collider(testdata, body_frame0, n_total, params):
    """Construct the headless WarpBodyCollider from extracted body geometry."""
    from _collision_warp import WarpBodyCollider
    idx = np.load(os.path.join(testdata, "body_tris_idx.npy"))
    return WarpBodyCollider(
        n_total=n_total, points_per_strand=int(params["pps"]),
        margin=float(params.get("collision_margin", 0.0005)),
        search_distance=float(params.get("collision_search", 0.003)),
        triangles=(body_frame0, idx),
    )


def run_from_testdata(testdata, out_path, start=None, end=None, overrides=None,
                      collision=False):
    """Convenience driver used by the CLI and the server.

    collision=True enables body collision: it seeds from the conditioned groom
    (roots 0.5 mm outside the collider) and feeds per-frame world-space body
    vertices to the unchanged Tokoya Warp collider.
    """
    groom_name = "groom_rest.npy"
    if collision and os.path.exists(
        os.path.join(testdata, "groom_rest_conditioned.npy")
    ):
        groom_name = "groom_rest_conditioned.npy"
    groom = np.load(os.path.join(testdata, groom_name))
    head = np.load(os.path.join(testdata, "head_world.npy"))
    params = default_params()
    if overrides:
        params.update(overrides)
    f0 = 0 if start is None else int(start)
    f1 = head.shape[0] if end is None else int(end)
    head = head[f0:f1]

    collider = None
    body_frames = None
    if collision:
        body_frames = np.load(
            os.path.join(testdata, "body_verts_world.npy"), mmap_mode="r"
        )[f0:f1]
        collider = _build_collider(
            testdata, np.asarray(body_frames[0]), groom.shape[0], params
        )

    out = simulate(groom, head, params,
                   progress=lambda i, n: print(f"  frame {i}/{n}", flush=True),
                   collider=collider, body_frames=body_frames)
    np.savez_compressed(out_path, positions=out, frame_start=f0, pps=params["pps"])
    finite = np.isfinite(out).all()
    moved = float(np.linalg.norm(out[-1] - out[0], axis=1).max())
    return {
        "out": out_path, "shape": list(out.shape),
        "finite": bool(finite), "max_tip_motion_m": moved,
        "collision": bool(collision), "groom": groom_name,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdata", default=os.path.join(_ROOT, "testdata"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "testdata", "hair_sim.npz"))
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--backend", default="CUDA", choices=["CUDA"])
    ap.add_argument("--collision", action="store_true",
                    help="enable body collision (requires CUDA backend)")
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    overrides = {"backend": args.backend}
    if args.collision:
        # Proven Tokoya body-collision settings (zero-penetration MCP run).
        overrides.update(backend="CUDA", substeps=8, iterations=20)
    if args.substeps is not None:
        overrides["substeps"] = args.substeps
    if args.iterations is not None:
        overrides["iterations"] = args.iterations

    import time
    t = time.time()
    r = run_from_testdata(args.testdata, args.out, args.start, args.end,
                          overrides=overrides, collision=args.collision)
    r["seconds"] = round(time.time() - t, 2)
    print(json.dumps(r, indent=2))
