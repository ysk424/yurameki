"""Validate the rigid head-follow model for hair roots.

Model:  root_world(f) = head_M(f) @ head_M(1)^-1 @ root_world(1)

Compares the prediction against Blender's actual evaluated root positions
(surface-deform result) at sample frames. If the scalp is rigidly skinned to
the head bone, the error is ~0 and the model is valid.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(HERE, "testdata")

head_world = np.load(os.path.join(TD, "head_world.npy"))          # (450,4,4)
groom_rest = np.load(os.path.join(TD, "groom_rest.npy"))          # (36000,3)
eval_roots = np.load(os.path.join(TD, "eval_roots_sample.npy"))   # (k,4000,3)
with open(os.path.join(TD, "eval_roots_sample.json")) as f:
    info = json.load(f)
frames = info["frames"]
pps = info["points_per_strand"]

roots1 = groom_rest[0::pps]            # (4000,3) rest roots at frame 1
H1_inv = np.linalg.inv(head_world[0])

# sanity: rest roots vs evaluated roots at frame 1
sanity = np.linalg.norm(roots1 - eval_roots[0], axis=1).max() * 1000
print(f"groom_rest vs eval frame1 : max {sanity:.4f} mm  (consistency)")

def apply(T, pts):
    h = np.column_stack([pts, np.ones(len(pts))])
    return (h @ T.T)[:, :3]

print(f"{'frame':>6} {'max mm':>10} {'mean mm':>10} {'95% mm':>10}")
for i, fr in enumerate(frames):
    T = head_world[fr - 1] @ H1_inv
    pred = apply(T, roots1)
    err = np.linalg.norm(pred - eval_roots[i], axis=1) * 1000.0
    print(f"{fr:>6} {err.max():>10.4f} {err.mean():>10.4f} {np.percentile(err,95):>10.4f}")
