"""Forward kinematics for the hip->head chain, Blender-independent (numpy only).

Input is the tsudura-equivalent clip:
  fk_channels: (frames, n_bones, 9)  -> loc(0:3) euler_xyz_rad(3:6) scale(6:9)
  meta:        chain_bone_names (root..head order), rest_local (4x4 per bone),
               inherit (parent per bone), head_bone
  armature_world: (frames, 4, 4)     -> the armature object's world matrix

We reconstruct each bone's pose matrix in armature space with Blender's rule:

    pose[root]  = rest[root] @ basis[root]
    pose[child] = pose[parent] @ (rest[parent]^-1 @ rest[child]) @ basis[child]

and the bone's world matrix is  armature_world @ pose[bone].

`basis` is the bone-local channel matrix  T(loc) @ R_xyz(euler) @ S(scale),
where Blender Euler order 'XYZ' means  v' = Rz @ Ry @ Rx @ v.
"""
from __future__ import annotations
import numpy as np


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_xyz_to_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Blender Euler order 'XYZ':  v' = Rz @ Ry @ Rx @ v."""
    return _rot_z(z) @ _rot_y(y) @ _rot_x(x)


def basis_matrix(loc, euler, scale) -> np.ndarray:
    """Bone-local channel matrix  T @ R @ S  (no shear)."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = euler_xyz_to_matrix(*euler) @ np.diag(np.asarray(scale, np.float64))
    m[:3, 3] = loc
    return m


def reconstruct_chain_world(fk_channels, meta, armature_world):
    """Return {bone_name: (frames,4,4) world matrices} for the whole chain."""
    chain = meta["chain_bone_names"]
    rest = {k: np.asarray(v, np.float64) for k, v in meta["rest_local"].items()}
    inherit = meta["inherit"]
    name_to_i = {n: i for i, n in enumerate(chain)}

    # rest-relative-to-parent, precomputed (constant across frames)
    rest_rel = {}
    parent = {}
    for n in chain:
        p = inherit[n]["parent"]
        p = p if (p in name_to_i) else None
        parent[n] = p
        rest_rel[n] = rest[n] if p is None else np.linalg.inv(rest[p]) @ rest[n]

    nf = fk_channels.shape[0]
    pose = {n: np.zeros((nf, 4, 4), np.float64) for n in chain}
    for f in range(nf):
        for i, n in enumerate(chain):
            basis = basis_matrix(
                fk_channels[f, i, 0:3],
                fk_channels[f, i, 3:6],
                fk_channels[f, i, 6:9],
            )
            p = parent[n]
            armspace = rest_rel[n] @ basis if p is None else pose[p][f] @ rest_rel[n] @ basis
            pose[n][f] = armspace

    return {n: armature_world @ pose[n] for n in chain}


def reconstruct_head_world(fk_channels, meta, armature_world) -> np.ndarray:
    """Return (frames,4,4) world matrices for the head bone."""
    worlds = reconstruct_chain_world(fk_channels, meta, armature_world)
    return worlds[meta["head_bone"]]
