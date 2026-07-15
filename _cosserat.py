"""Stable Cosserat Rods elastic-rod solver core for Yurameki2.

This module replaces the previous XPBD distance/bend constraint solve plus the
FK "KEEP LENGTH" reconnection with a genuine Cosserat elastic-rod solver.

Each strand is a chain of vertices ``p_0 .. p_{pps-1}`` with one unit quaternion
material frame ``q_e`` per segment ``e = 0 .. pps-2``. The third director
``d3(q_e) = R(q_e) * e3`` is the segment tangent. Two energies act on the rod:

* stretch/shear, per segment, coupling positions to the director frame::

      C_s(e) = (p_{e+1} - p_e) - l_e * d3(q_e)

  When ``C_s = 0`` the segment has exactly rest length ``l_e`` *and* points along
  its frame tangent, so inextensibility holds without any FK reconnection.

* bend/twist, per interior segment pair ``(e, e+1)``, as a Darboux residual::

      C_b(e) = Im(conj(q_e) * q_{e+1}) - Omega0_e

The solver follows the Stable Cosserat Rods split: a Vertex-Block-Descent
position solve (fixed orientations, exact scalar Hessian) alternated with a
quasi-static local Gauss-Newton orientation solve (rotational inertia treated as
negligible for thin rods). Both passes are local and use red/black colouring so
one Warp launch never has two threads writing the same vertex or segment.

The orientation Gauss-Newton derivatives use the body-frame right-multiply
convention ``q <- q * exp(omega)`` (see ``_exp_quat``); every derivative in
``cosserat_orientation_kernel`` is consistent with that convention and is checked
by finite differences in ``tools/validate_cosserat.py``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

E3 = wp.constant(wp.vec3(0.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Device helper functions
# --------------------------------------------------------------------------- #
@wp.func
def _quat_conj(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def _quat_negate(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], -q[3])


@wp.func
def _dir3(q: wp.quat) -> wp.vec3:
    # Third material director = tangent = R(q) * e3.
    return wp.quat_rotate(q, E3)


@wp.func
def _skew(v: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.func
def _exp_quat(w: wp.vec3) -> wp.quat:
    # Body-frame incremental rotation q_delta with rotation vector w.
    theta = wp.length(w)
    if theta < 1.0e-8:
        return wp.normalize(wp.quat(0.5 * w[0], 0.5 * w[1], 0.5 * w[2], 1.0))
    half = 0.5 * theta
    s = wp.sin(half) / theta
    return wp.quat(w[0] * s, w[1] * s, w[2] * s, wp.cos(half))


@wp.func
def _rel_darboux(qa: wp.quat, qb: wp.quat) -> wp.quat:
    # Relative rotation conj(qa) * qb, hemisphere-normalised so the scalar part
    # is non-negative (rotation angle < 180 deg). The same rule is used at rest
    # (numpy init) and at runtime so the Darboux residual is well defined.
    d = _quat_conj(qa) * qb
    if d[3] < 0.0:
        d = _quat_negate(d)
    return d


# --------------------------------------------------------------------------- #
# Position pass: Vertex Block Descent, orientations held fixed.
# --------------------------------------------------------------------------- #
@wp.kernel
def cosserat_position_kernel(
    predicted: wp.array(dtype=wp.vec3),
    inertial: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    orient: wp.array(dtype=wp.quat),
    seg_rest: wp.array(dtype=float),
    points_per_strand: int,
    parity: int,
    k_ss: float,
    inv_dt2: float,
    relaxation: float,
):
    i = wp.tid()
    w = inv_mass[i]
    if w <= 0.0:
        return  # kinematic vertex, pinned by the predict kernel

    local = i % points_per_strand
    if local % 2 != parity:
        return

    strand = i // points_per_strand
    segs = points_per_strand - 1

    mass = 1.0 / w
    c_inertia = mass * inv_dt2
    grad = c_inertia * (predicted[i] - inertial[i])
    h = c_inertia

    # Right segment e = local (this vertex is p_e). dC/dp_e = -I.
    if local < segs:
        e = strand * segs + local
        d3 = _dir3(orient[e])
        c_s = (predicted[i + 1] - predicted[i]) - seg_rest[e] * d3
        grad = grad - k_ss * c_s
        h = h + k_ss

    # Left segment e = local-1 (this vertex is p_{e+1}). dC/dp_{e+1} = +I.
    if local > 0:
        e = strand * segs + (local - 1)
        d3 = _dir3(orient[e])
        c_s = (predicted[i] - predicted[i - 1]) - seg_rest[e] * d3
        grad = grad + k_ss * c_s
        h = h + k_ss

    if h <= 0.0:
        return
    predicted[i] = predicted[i] - (relaxation / h) * grad


@wp.kernel
def cosserat_position_tridiagonal_kernel(
    predicted: wp.array(dtype=wp.vec3),
    inertial: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    orient: wp.array(dtype=wp.quat),
    seg_rest: wp.array(dtype=float),
    scratch_c: wp.array(dtype=float),
    scratch_d: wp.array(dtype=wp.vec3),
    points_per_strand: int,
    k_ss: float,
    inv_dt2: float,
):
    """Exact position solve for one strand (orientations fixed).

    With the orientations frozen, the stretch/shear position objective is a
    quadratic whose Hessian is scalar-tridiagonal along the chain, so a single
    Thomas sweep per strand is an *exact* minimiser -- the rod stays at rest
    length regardless of how stiff ``k_ss`` is relative to inertia. This is what
    makes the FK "keep length" reconnection unnecessary. Assumes the locked
    (kinematic) vertices form a contiguous prefix at the strand root.
    """
    sid = wp.tid()
    pps = points_per_strand
    base = sid * pps
    segs = pps - 1

    # first free (non-kinematic) vertex
    f0 = int(0)
    while f0 < pps and inv_mass[base + f0] <= 0.0:
        f0 += 1
    if f0 >= pps:
        return  # whole strand is kinematic

    # forward sweep over free rows r = f0 .. pps-1
    r = f0
    while r < pps:
        gi = base + r
        has_left = r > 0
        has_right = r < pps - 1
        mass = 1.0 / inv_mass[gi]
        diag = mass * inv_dt2
        rhs = mass * inv_dt2 * inertial[gi]
        if has_right:
            diag += k_ss
            se = sid * segs + r
            rhs = rhs - k_ss * seg_rest[se] * _dir3(orient[se])
        if has_left:
            diag += k_ss
            se = sid * segs + (r - 1)
            rhs = rhs + k_ss * seg_rest[se] * _dir3(orient[se])

        sub = float(0.0)
        if has_left and r > f0:
            sub = -k_ss  # couples to previous free row
        elif has_left and r == f0:
            # previous vertex is kinematic: fold its fixed position into rhs
            rhs = rhs + k_ss * predicted[gi - 1]

        sup = float(0.0)
        if has_right:
            sup = -k_ss

        denom = diag
        if r > f0:
            denom = diag - sub * scratch_c[gi - 1]
        inv_denom = 1.0 / denom
        scratch_c[gi] = sup * inv_denom
        if r == f0:
            scratch_d[gi] = rhs * inv_denom
        else:
            scratch_d[gi] = (rhs - sub * scratch_d[gi - 1]) * inv_denom
        r += 1

    # back substitution
    predicted[base + pps - 1] = scratch_d[base + pps - 1]
    r = pps - 2
    while r >= f0:
        gi = base + r
        predicted[gi] = scratch_d[gi] - scratch_c[gi] * predicted[gi + 1]
        r -= 1


# --------------------------------------------------------------------------- #
# Orientation pass: quasi-static local Gauss-Newton, positions held fixed.
# --------------------------------------------------------------------------- #
@wp.kernel
def cosserat_orientation_kernel(
    predicted: wp.array(dtype=wp.vec3),
    orient: wp.array(dtype=wp.quat),
    darboux_rest: wp.array(dtype=wp.vec3),
    seg_rest: wp.array(dtype=float),
    points_per_strand: int,
    parity: int,
    k_ss: float,
    k_bt: float,
    damping: float,
    relaxation: float,
    max_omega: float,
    bend_root_boost: float,
):
    e = wp.tid()
    segs = points_per_strand - 1
    local_seg = e % segs
    if local_seg % 2 != parity:
        return

    strand = e // segs
    i = strand * points_per_strand + local_seg
    j = i + 1
    q = orient[e]
    # Option 1: stiffen bending toward the root (linear ramp) where the
    # cantilever bending moment is highest, so curvature distributes over several
    # joints instead of concentrating -- and buckling -- at the first free one.
    k_bt_eff = k_bt * (1.0 + bend_root_boost * (1.0 - float(local_seg) / float(segs)))

    a = wp.mat33(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )
    rhs = wp.vec3(0.0, 0.0, 0.0)
    ident = wp.identity(n=3, dtype=float)

    # --- stretch / shear: s = (p_j - p_i) - l * d3(q) ----------------------- #
    d3 = _dir3(q)
    seg_len = seg_rest[e]
    s = (predicted[j] - predicted[i]) - seg_len * d3
    # d(d3)/d(omega) columns are R(q) * (e_k x e3):  (0,-1,0),(1,0,0),(0,0,0).
    dd0 = wp.quat_rotate(q, wp.vec3(0.0, -1.0, 0.0))
    dd1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    # Js = d(s)/d(omega) = -l * d(d3)/d(omega); third column is zero.
    js = wp.mat33(
        -seg_len * dd0[0], -seg_len * dd1[0], 0.0,
        -seg_len * dd0[1], -seg_len * dd1[1], 0.0,
        -seg_len * dd0[2], -seg_len * dd1[2], 0.0,
    )
    jst = wp.transpose(js)
    a = a + k_ss * (jst * js)
    rhs = rhs - k_ss * (jst * s)

    # --- bend / twist toward the next segment (pair e, e+1) ----------------- #
    if local_seg < segs - 1:
        qn = orient[e + 1]
        d = _rel_darboux(q, qn)
        dv = wp.vec3(d[0], d[1], d[2])
        dw = d[3]
        b_r = dv - darboux_rest[strand * (segs - 1) + local_seg]
        # d(Im(conj(q) qn))/d(omega) = 0.5 * (-dw I + skew(dv)) for q <- q*exp(w)
        jr = 0.5 * (-dw * ident + _skew(dv))
        jrt = wp.transpose(jr)
        a = a + k_bt_eff * (jrt * jr)
        rhs = rhs - k_bt_eff * (jrt * b_r)

    # --- bend / twist toward the previous segment (pair e-1, e) ------------- #
    if local_seg > 0:
        qp = orient[e - 1]
        g = _rel_darboux(qp, q)
        gv = wp.vec3(g[0], g[1], g[2])
        gw = g[3]
        b_l = gv - darboux_rest[strand * (segs - 1) + (local_seg - 1)]
        # d(Im(conj(qp) q))/d(omega) = 0.5 * (gw I + skew(gv)) for q <- q*exp(w)
        jl = 0.5 * (gw * ident + _skew(gv))
        jlt = wp.transpose(jl)
        a = a + k_bt_eff * (jlt * jl)
        rhs = rhs - k_bt_eff * (jlt * b_l)

    # --- damped Gauss-Newton solve and body-frame update -------------------- #
    a = a + damping * ident
    det = wp.determinant(a)
    if wp.abs(det) < 1.0e-20:
        return
    omega = relaxation * (wp.inverse(a) * rhs)

    mag = wp.length(omega)
    if mag > max_omega and mag > 1.0e-9:
        omega = omega * (max_omega / mag)

    orient[e] = wp.normalize(q * _exp_quat(omega))


# --------------------------------------------------------------------------- #
# Buckling resistance: position-space angle constraint (anti fold-back).
# --------------------------------------------------------------------------- #
@wp.kernel
def cosserat_buckle_kernel(
    predicted: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    points_per_strand: int,
    color: int,
    cos_threshold: float,
    stiffness: float,
):
    """Option 2: a one-sided angle constraint on three consecutive points that
    resists a strand folding back on itself. Active only when the bend cosine
    ``cos(theta) = t_hat_i . t_hat_{i+1}`` drops below ``cos_threshold``; it then
    applies a length-preserving (perpendicular) position correction that opens
    the angle. Uses the position-space cosine directly (a dot product -- no acos,
    no lookup table). 3-colour over the joint index so one launch never writes a
    point that another thread in the same launch also writes.
    """
    tid = wp.tid()
    interior = points_per_strand - 2
    if interior <= 0:
        return
    strand = tid // interior
    local = tid % interior              # 0 .. pps-3  (joint at point local+1)
    if (local + 1) % 3 != color:
        return
    ci = strand * points_per_strand + local + 1
    w0 = inv_mass[ci - 1]
    w1 = inv_mass[ci]
    w2 = inv_mass[ci + 1]
    if w0 + w1 + w2 <= 0.0:
        return
    p0 = predicted[ci - 1]
    p1 = predicted[ci]
    p2 = predicted[ci + 1]
    a = p1 - p0
    b = p2 - p1
    la = wp.length(a)
    lb = wp.length(b)
    if la < 1.0e-9 or lb < 1.0e-9:
        return
    ah = a / la
    bh = b / lb
    cosv = wp.dot(ah, bh)
    if cosv >= cos_threshold:
        return  # not folding past the threshold -- leave the natural bend alone
    # C = cos_threshold - cos (> 0 when folding); gradients wrt the three points.
    # d(cos)/da and d(cos)/db are perpendicular to a,b, so the fix preserves length.
    dcos_da = (bh - cosv * ah) / la
    dcos_db = (ah - cosv * bh) / lb
    g0 = dcos_da
    g1 = dcos_db - dcos_da
    g2 = -dcos_db
    denom = w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1) + w2 * wp.dot(g2, g2)
    if denom < 1.0e-12:
        return
    dl = -(cos_threshold - cosv) / denom
    predicted[ci - 1] = p0 + (stiffness * w0 * dl) * g0
    predicted[ci] = p1 + (stiffness * w1 * dl) * g1
    predicted[ci + 1] = p2 + (stiffness * w2 * dl) * g2


# --------------------------------------------------------------------------- #
# Strain-rate (internal viscosity) velocity damping.
# --------------------------------------------------------------------------- #
@wp.kernel
def _internal_damp_kernel(
    vel_in: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    points_per_strand: int,
    mu: float,
):
    """Velocity Laplacian along each strand between free joints (a viscoelastic
    rod's internal viscosity). It damps the internal *deformation* velocity --
    ringing, jitter, and frizz -- while preserving the bulk motion that follows
    the kinematic root and gravity: a spatially smooth (rigid/translational)
    velocity field is a fixed point of the Laplacian, so only the high-frequency
    part decays. Only free-free joint pairs are coupled, so a strand is never
    damped toward the stored-zero kinematic root velocity.

    Jacobi form (reads the snapshot ``vel_in``, writes ``vel_out``): because each
    free-free edge contributes equal and opposite corrections to its two joints,
    the per-strand mean velocity -- the bulk momentum -- is exactly preserved.
    """
    i = wp.tid()
    if inv_mass[i] <= 0.0:
        vel_out[i] = vel_in[i]
        return
    local = i % points_per_strand
    lap = wp.vec3(0.0, 0.0, 0.0)
    if local > 0 and inv_mass[i - 1] > 0.0:
        lap = lap + (vel_in[i - 1] - vel_in[i])
    if local < points_per_strand - 1 and inv_mass[i + 1] > 0.0:
        lap = lap + (vel_in[i + 1] - vel_in[i])
    vel_out[i] = vel_in[i] + mu * lap


# --------------------------------------------------------------------------- #
# Host-side initialisation (parallel-transport frames from rest geometry).
# --------------------------------------------------------------------------- #
def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit quaternion (xyzw) rotating unit vector ``a`` onto unit vector ``b``."""
    c = float(np.dot(a, b))
    if c > 1.0 - 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if c < -1.0 + 1.0e-8:
        # 180 degrees: rotate about any axis orthogonal to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.dot(axis, axis) < 1.0e-12:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + c], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_conj_np(a: np.ndarray) -> np.ndarray:
    return np.array([-a[0], -a[1], -a[2], a[3]], dtype=np.float64)


def _rel_darboux_np(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    d = _quat_mul(_quat_conj_np(qa), qb)
    if d[3] < 0.0:
        d = -d
    return d


def init_cosserat(points: np.ndarray, pps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-segment rest frames, rest Darboux vectors and rest lengths.

    Parameters
    ----------
    points : (n_strands * pps, 3) rest positions.
    pps    : points per strand.

    Returns
    -------
    quats        : (n_strands * (pps-1), 4) float32 xyzw segment frames.
    darboux_rest : (n_strands * (pps-2), 3) float32 rest Darboux vectors.
    seg_rest     : (n_strands * (pps-1),) float32 rest segment lengths.
    """
    pps = int(pps)
    strands = np.asarray(points, dtype=np.float64).reshape(-1, pps, 3)
    n_strands = strands.shape[0]
    segs = pps - 1
    bends = max(pps - 2, 0)

    quats = np.zeros((n_strands, segs, 4), dtype=np.float64)
    darboux = np.zeros((n_strands, bends, 3), dtype=np.float64)
    seg_rest = np.zeros((n_strands, segs), dtype=np.float64)

    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for s in range(n_strands):
        prev_tangent = None
        prev_q = None
        for e in range(segs):
            d = strands[s, e + 1] - strands[s, e]
            length = float(np.linalg.norm(d))
            seg_rest[s, e] = max(length, 1.0e-7)
            tangent = d / length if length > 1.0e-9 else ref.copy()

            if e == 0:
                q = _shortest_arc(ref, tangent)
            else:
                # Parallel transport: rotate the previous frame so its tangent
                # tracks the new segment tangent (bishop / zero-twist frame).
                delta = _shortest_arc(prev_tangent, tangent)
                q = _quat_mul(delta, prev_q)
            q = q / np.linalg.norm(q)
            quats[s, e] = q
            prev_tangent = tangent
            prev_q = q

        for e in range(bends):
            darboux[s, e] = _rel_darboux_np(quats[s, e], quats[s, e + 1])[:3]

    return (
        np.ascontiguousarray(quats.reshape(-1, 4), dtype=np.float32),
        np.ascontiguousarray(darboux.reshape(-1, 3), dtype=np.float32),
        np.ascontiguousarray(seg_rest.reshape(-1), dtype=np.float32),
    )
