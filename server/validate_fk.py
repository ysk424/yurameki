"""Validate the numpy FK reconstruction against Blender's ground-truth head world.

Proves the server can rebuild the head bone's world motion from the
tsudura-equivalent FK clip alone (no Blender), which is what drives the hair
roots.
"""
import json, os
import numpy as np
from fk import reconstruct_head_world

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(HERE, "testdata")

fk_channels = np.load(os.path.join(TD, "fk_channels.npy"))
armature_world = np.load(os.path.join(TD, "armature_world.npy"))
ground_truth = np.load(os.path.join(TD, "head_world.npy"))
with open(os.path.join(TD, "fk_meta.json"), encoding="utf-8") as f:
    meta = json.load(f)

recon = reconstruct_head_world(fk_channels, meta, armature_world)

diff = recon - ground_truth
max_err = np.abs(diff).max()
# translation error in metres per frame
trans_err = np.linalg.norm(recon[:, :3, 3] - ground_truth[:, :3, 3], axis=1)
# rotation error: angle between reconstructed and gt rotation, degrees
def _orthonormal(m):
    r = m[:3, :3].astype(np.float64).copy()
    for k in range(3):
        n = np.linalg.norm(r[:, k])
        if n > 0:
            r[:, k] /= n
    return r

def rot_angle_deg(a, b):
    # strip the baked 0.01 world scale before measuring the rotation angle
    r = _orthonormal(a) @ _orthonormal(b).T
    c = (np.trace(r) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
rot_err = np.array([rot_angle_deg(recon[f], ground_truth[f]) for f in range(len(recon))])

print(f"frames                : {len(recon)}")
print(f"max abs matrix error  : {max_err:.3e}")
print(f"max translation error : {trans_err.max()*1000:.4f} mm  (mean {trans_err.mean()*1000:.4f} mm)")
print(f"max rotation error    : {rot_err.max():.4e} deg  (mean {rot_err.mean():.4e} deg)")
print(f"frame1 head pos (recon): {recon[0,:3,3]}")
print(f"frame1 head pos (truth): {ground_truth[0,:3,3]}")
ok = max_err < 1e-6
print("RESULT:", "PASS" if ok else "FAIL")
